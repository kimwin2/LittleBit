"""
Step 2: Train 0.9-bit Residual Model via QAT (Matryoshka Style)

Loads the original FP model and the trained Step 1 draft (0.1-bit) model.
Computes residual weights (W_original - W_draft_approx) and initializes
a 0.9-bit model from those residuals. The 0.1-bit model is frozen;
only the 0.9-bit residual parameters are trained.

At inference time, the combined model (0.1-bit + 0.9-bit) approximates the
original 1.0-bit model for use as the target in speculative decoding.

Usage:
    deepspeed --num_gpus=4 train_step2_residual.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --draft_model_path outputs/step1_draft_0.1bit/<timestamp> \
        --dataset wikitext2_sharegpt \
        --eff_bit 0.9 \
        --ds_config_path configs/zero3.json \
        --save_dir outputs/step2_residual_0.9bit
"""

import re
import argparse
import datetime
import json
import os
import gc
from pathlib import Path
from copy import deepcopy

import deepspeed
import GPUtil
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import default_data_collator
from transformers import AutoModelForCausalLM, AutoConfig, TrainingArguments, set_seed, Trainer
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from quantization.utils import apply_littlebit_patch
from quantization.utils.quant_util import load_quantized_model, get_quant_func_and_mod
from quantization.modules import LittleBitLinear
from utils.datautils import prepare_dataset, load_tokenizer
from utils.misc import setup_logger
from utils.utils import prepare_model_for_training, print_trainable_parameters

logger = setup_logger(__name__)


def get_device_config():
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None, None
    device_map = "auto"
    local_rank_str = os.environ.get('LOCAL_RANK')
    if local_rank_str is not None:
        try:
            local_rank = int(local_rank_str)
            device_map = {'': local_rank}
        except ValueError:
            pass
    return len(gpus), device_map


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception(f'Boolean value expected: {value}')


def get_args():
    parser = argparse.ArgumentParser(description="Step 2: Train 0.9-bit Residual Model")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft_model_path", type=str, required=True,
                        help="Path to trained Step 1 draft model (0.1-bit)")
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--dataset", type=str, default="wikitext2_sharegpt",
                        choices=['c4', 'wikitext2', 'c4_wiki', 'wikitext2_sharegpt'])
    parser.add_argument("--sharegpt_path", type=str, default=None,
                        help="Path to local ShareGPT .jsonl file (optional)")
    parser.add_argument("--save_dir", type=str, default='outputs/step2_residual_0.9bit')
    parser.add_argument("--f_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--l2l_loss_scale", type=float, default=10.0)
    parser.add_argument("--dataset_prepared", type=str2bool, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ds_config_path", type=str, default="configs/zero3.json")
    parser.add_argument("--exp_name", type=str, default="LittleBit_Step2_Residual")
    parser.add_argument("--run_name", type=str, default="step2_residual_0.9bit")
    parser.add_argument("--report", nargs="+", default=["wandb"], choices=["wandb", "tensorboard"])
    parser.add_argument("--quant_func", type=str, default="STEBinary")
    parser.add_argument("--quant_mod", type=str, default="LittleBitLinear")

    # Quantization parameters - 0.9 bit for residual model
    parser.add_argument("--residual", type=str2bool, default=False)
    parser.add_argument("--split_dim", type=int, default=1024)
    parser.add_argument("--eff_bit", type=float, default=0.9)
    parser.add_argument("--kv_factor", type=float, default=1.0)
    parser.add_argument("--min_split_dim", type=int, default=8)

    args = parser.parse_args()
    return args


class MatryoshkaResidualModel(nn.Module):
    """
    Wrapper model that combines a frozen 0.1-bit draft model with a trainable 0.9-bit residual model.
    
    Forward pass: logits = draft_logits + residual_logits
    Only the residual model parameters are trainable.
    """
    def __init__(self, draft_model, residual_model, config):
        super().__init__()
        self.draft_model = draft_model
        self.residual_model = residual_model
        self._config = config
        
        # Freeze draft model completely
        for param in self.draft_model.parameters():
            param.requires_grad = False
        self.draft_model.eval()
    
    @property
    def config(self):
        return self._config
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, 
                output_hidden_states=False, **kwargs):
        # Draft forward (frozen, no grad)
        with torch.no_grad():
            draft_outputs = self.draft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
        
        # Residual forward (trainable)
        residual_outputs = self.residual_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        # Combine logits
        combined_logits = draft_outputs.logits + residual_outputs.logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = combined_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Build output dict compatible with HuggingFace
        from transformers.modeling_outputs import CausalLMOutputWithPast
        
        combined_hidden_states = None
        if output_hidden_states:
            # Combine hidden states from both models
            draft_hs = draft_outputs.hidden_states
            residual_hs = residual_outputs.hidden_states
            combined_hidden_states = tuple(
                d + r for d, r in zip(draft_hs, residual_hs)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=combined_logits,
            past_key_values=None,
            hidden_states=combined_hidden_states,
            attentions=None,
        )
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.residual_model.gradient_checkpointing_enable(**kwargs)
    
    def get_input_embeddings(self):
        return self.residual_model.get_input_embeddings()
    
    def parameters(self, recurse=True):
        # Only return residual model parameters for optimizer
        return self.residual_model.parameters(recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        return self.residual_model.named_parameters(prefix=prefix + 'residual_model.', recurse=recurse)


class MatryoshkaKDTrainer(Trainer):
    """
    Knowledge distillation trainer for the Matryoshka residual model.
    The combined model (draft + residual) is trained to match the FP teacher.
    """
    def __init__(self, teacher_model, l2l_loss_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.l2l_loss_scale = l2l_loss_scale

    def ce_loss(self, student_logits, teacher_logits):
        model_output_log_prob = F.log_softmax(student_logits, dim=-1)
        real_output_soft = F.softmax(teacher_logits, dim=-1)
        return F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")

    def mse_loss(self, student_logits, teacher_logits):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return F.mse_loss(student_logits, teacher_logits)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["output_hidden_states"] = True

        # Teacher inference
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        teacher_logits = teacher_outputs.get("logits")
        teacher_reps = teacher_outputs.hidden_states[1:]
        del teacher_outputs

        # Combined model (draft frozen + residual trainable)
        outputs = model(**inputs)

        student_logits = outputs.logits
        student_reps = outputs.hidden_states[1:] if outputs.hidden_states else []

        if not return_outputs:
            del_outputs = outputs

        # KD loss on logits
        kd_loss = self.ce_loss(student_logits, teacher_logits)

        # Layer-to-layer loss
        l2l_loss = torch.tensor(0.0, device=student_logits.device)
        if student_reps and teacher_reps:
            for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                tmp_loss = self.mse_loss(student_rep, teacher_rep)
                l2l_loss = l2l_loss + tmp_loss
            l2l_loss = self.l2l_loss_scale * l2l_loss

        loss = kd_loss + l2l_loss

        self.log({
            "l2l_loss": l2l_loss.item(),
            "kd_loss": kd_loss.item(),
        })

        return (loss, outputs) if return_outputs else loss


def compute_residual_weights(original_model, draft_model, device='cpu'):
    """
    Compute residual weights: W_residual = W_original - W_draft_approx
    
    For each layer that has been quantized in the draft model (LittleBitLinear),
    reconstruct the approximation and subtract from the original weight.
    """
    residual_weights = {}
    
    # Get original model state dict (full precision weights)
    original_state_dict = original_model.state_dict()
    
    # Iterate through draft model to find LittleBitLinear modules and compute residual
    quant_func_name = "STEBinary"
    from quantization.functions import STEBinary
    
    draft_calc_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    for name, module in draft_model.named_modules():
        if isinstance(module, LittleBitLinear):
            # Reconstruct the 0.1-bit approximation:
            # W_approx = (U * (u1^T @ u2)) @ (V * (v1^T @ v2))
            # where U, V are binarized
            U = module.U.data.float().to(draft_calc_device)
            V = module.V.data.float().to(draft_calc_device)
            u1 = module.u1.data.float().to(draft_calc_device)
            u2 = module.u2.data.float().to(draft_calc_device)
            v1 = module.v1.data.float().to(draft_calc_device)
            v2 = module.v2.data.float().to(draft_calc_device)
            
            # Binarize U and V (sign function)
            if module._binarized:
                Uq = U  # Already binarized
                Vq = V
            else:
                Uq = STEBinary(U)
                Vq = STEBinary(V)
            
            # Reconstruct: W_approx = ((Uq * (u1^T @ u2)) @ (Vq * (v1^T @ v2)))
            # Following forward: ((((x * v2) @ Vq^T) * (v1 * u2)) @ Uq^T) * u1
            # This means W_approx^T = diag(u1) @ Uq @ diag(v1*u2) @ Vq @ diag(v2)
            # So W_approx = diag(v2) @ Vq^T @ diag(v1*u2) @ Uq^T @ diag(u1)
            # But more directly:
            W_approx = (Uq * (u1.t() @ u2)) @ (Vq * (v1.t() @ v2))
            
            # Find the original weight key
            # name is like "model.layers.0.self_attn.q_proj"
            weight_key = name + ".weight"
            if weight_key in original_state_dict:
                W_original = original_state_dict[weight_key].float().to(draft_calc_device)
                W_residual = W_original - W_approx
                residual_weights[weight_key] = W_residual.cpu()
                logger.info(f"Computed residual for {weight_key}: "
                           f"original_norm={W_original.norm():.4f}, "
                           f"approx_norm={W_approx.norm():.4f}, "
                           f"residual_norm={W_residual.norm():.4f}")
                del W_original, W_approx
            else:
                logger.warning(f"Original weight key {weight_key} not found, skipping")
            
            del U, V, u1, u2, v1, v2, Uq, Vq
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return residual_weights


def initialize_residual_model_weights(residual_model, residual_weights):
    """
    Initialize the residual model's LittleBitLinear layers from pre-computed residual weights.
    
    This replaces the default SVD initialization with one based on:
    W_residual = W_original - W_0.1bit_approx
    """
    from quantization.functions import STEBinary
    
    calc_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    for name, module in residual_model.named_modules():
        if isinstance(module, LittleBitLinear):
            weight_key = name + ".weight"
            if weight_key in residual_weights:
                W_residual = residual_weights[weight_key].float()
                
                logger.info(f"Re-initializing {name} from residual weights (shape={W_residual.shape})")
                
                # Perform SVD decomposition on residual weights
                W_calc = W_residual.to(calc_device)
                split_dim = module.split_dim
                
                U_t, S_t, V_t = torch.svd_lowrank(W_calc, q=split_dim)
                Vh_t = V_t.t()
                
                sqrt_S = torch.sqrt(torch.diag(S_t))[:, :split_dim]
                
                U_new = (U_t @ sqrt_S).contiguous()
                V_new = (sqrt_S.t() @ Vh_t).contiguous()
                
                # Rank-one decomposition for scale factors
                def rank_one_decompose(X):
                    U_r, S_r, V_r = torch.svd_lowrank(X.to(calc_device), q=1)
                    Vh_r = V_r.t()
                    sqrt_S0 = torch.sqrt(S_r[0])
                    u_comp = (U_r[:, :1] * sqrt_S0).t().contiguous()
                    v_comp = (sqrt_S0 * Vh_r[:1, :]).contiguous()
                    return u_comp, v_comp
                
                v1_new, v2_new = rank_one_decompose(torch.abs(V_new))
                u1_new, u2_new = rank_one_decompose(torch.abs(U_new))
                
                dtype = module.U.dtype if hasattr(module, 'U') and module.U is not None else torch.bfloat16
                device = 'cpu'
                
                module.U = nn.Parameter(U_new.to(device=device, dtype=dtype), requires_grad=True)
                module.V = nn.Parameter(V_new.to(device=device, dtype=dtype), requires_grad=True)
                module.u1 = nn.Parameter(u1_new.to(device=device, dtype=dtype), requires_grad=True)
                module.u2 = nn.Parameter(u2_new.to(device=device, dtype=dtype), requires_grad=True)
                module.v1 = nn.Parameter(v1_new.to(device=device, dtype=dtype), requires_grad=True)
                module.v2 = nn.Parameter(v2_new.to(device=device, dtype=dtype), requires_grad=True)
                
                del W_calc, U_t, S_t, V_t, Vh_t, U_new, V_new
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                logger.info(f"  -> Initialized {name} with split_dim={split_dim}")
            else:
                logger.warning(f"No residual weight found for {name}, keeping default init")


def get_save_dir(args):
    f_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if args.f_name is None else args.f_name
    save_dir = os.path.join(args.save_dir, f_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def get_training_arguments(args, save_dir):
    return TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        save_steps=10000,
        output_dir=save_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        deepspeed=args.ds_config_path,
        report_to=args.report,
        run_name=args.run_name,
    )


def load_teacher_model(args, num_gpus, torch_dtype, config_path="configs/zero3_inference.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    _ = HfDeepSpeedConfig(config)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.config.use_cache = False
    teacher_model, _, _, _ = deepspeed.initialize(
        model=teacher_model, model_parameters=teacher_model.parameters(), config=config,
    )
    return teacher_model


def save_residual_artifacts(trainer, residual_model, tokenizer, save_dir, args):
    """Save only the residual (0.9-bit) model weights."""
    try:
        logger.info("Saving residual model artifacts...")

        if hasattr(trainer, 'accelerator'):
            unwrapped = trainer.accelerator.unwrap_model(residual_model)
        else:
            unwrapped = residual_model
            while hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module
        
        # If it's a MatryoshkaResidualModel, extract just the residual part
        if isinstance(unwrapped, MatryoshkaResidualModel):
            actual_residual = unwrapped.residual_model
        else:
            actual_residual = unwrapped

        use_ds = (args.ds_config_path is not None)
        final_cpu_state_dict = {}

        if use_ds:
            LAYER_CHUNK_SIZE = 4
            for name, module in actual_residual.named_children():
                if isinstance(module, torch.nn.ModuleList):
                    num_layers = len(module)
                    for i in range(0, num_layers, LAYER_CHUNK_SIZE):
                        end_idx = min(i + LAYER_CHUNK_SIZE, num_layers)
                        layer_group = module[i:end_idx]
                        logger.info(f"Gathering layers {i} to {end_idx-1}...")
                        with deepspeed.zero.GatheredParameters(layer_group.parameters(), modifier_rank=0):
                            if args.local_rank == 0 or args.local_rank == -1:
                                for idx, layer in enumerate(layer_group):
                                    layer_global_idx = i + idx
                                    layer_state_dict = layer.state_dict()
                                    for k, v in layer_state_dict.items():
                                        final_cpu_state_dict[f"{name}.{layer_global_idx}.{k}"] = v.cpu()
                else:
                    logger.info(f"Processing module: {name}")
                    with deepspeed.zero.GatheredParameters(module.parameters(), modifier_rank=0):
                        if args.local_rank == 0 or args.local_rank == -1:
                            module_state_dict = module.state_dict()
                            for k, v in module_state_dict.items():
                                final_cpu_state_dict[f"{name}.{k}"] = v.cpu()

            remaining_params = [p for n, p in actual_residual.named_parameters() if '.' not in n]
            if remaining_params:
                with deepspeed.zero.GatheredParameters(remaining_params, modifier_rank=0):
                    if args.local_rank == 0 or args.local_rank == -1:
                        for n, p in actual_residual.named_parameters():
                            if '.' not in n:
                                final_cpu_state_dict[n] = p.cpu()
        else:
            final_cpu_state_dict = {k: v.cpu() for k, v in actual_residual.state_dict().items()}

        if args.local_rank == 0 or args.local_rank == -1:
            quant_params = {
                "quant_func": args.quant_func,
                "eff_bit": args.eff_bit,
                "split_dim": args.split_dim,
                "residual": args.residual,
                "kv_factor": args.kv_factor,
                "min_split_dim": args.min_split_dim,
                "quant_mod": args.quant_mod,
                "matryoshka_stage": "residual",
                "matryoshka_bit": 0.9,
                "draft_model_path": args.draft_model_path,
            }

            littlebit_config_path = os.path.join(save_dir, "littlebit_config.json")
            with open(littlebit_config_path, "w", encoding="utf-8") as f:
                json.dump(quant_params, f, indent=2)

            for key, value in quant_params.items():
                if key not in ("draft_model_path", "matryoshka_stage", "matryoshka_bit"):
                    setattr(actual_residual.config, key, value)

            actual_residual.config.use_cache = True

            for k, v in final_cpu_state_dict.items():
                if "packed" not in k and "shape" not in k and v.dtype == torch.float32:
                    final_cpu_state_dict[k] = v.to(torch.bfloat16)

            actual_residual.save_pretrained(save_dir, state_dict=final_cpu_state_dict, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)

            logger.info(f"Residual model (0.9-bit) saved to {save_dir}")
            del final_cpu_state_dict
            gc.collect()

    except Exception as save_err:
        logger.error(f"Failed during save: {save_err}", exc_info=True)


def main():
    args = get_args()
    set_seed(args.seed)

    save_dir = get_save_dir(args)
    num_gpus, device_map = get_device_config()

    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_id)

    logger.info(f"Preparing training data ({args.dataset})...")
    datasets = prepare_dataset(args, tokenizer)

    # ===== Step A: Load the original FP model =====
    logger.info("Loading original FP model for residual computation...")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float32, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )

    # ===== Step B: Load the trained 0.1-bit draft model =====
    logger.info(f"Loading trained 0.1-bit draft model from {args.draft_model_path}...")
    
    # Load draft model config
    draft_config_path = os.path.join(args.draft_model_path, "littlebit_config.json")
    with open(draft_config_path, 'r') as f:
        draft_config = json.load(f)
    
    draft_args = argparse.Namespace(
        quant_func=draft_config.get("quant_func", "STEBinary"),
        quant_mod=draft_config.get("quant_mod", "LittleBitLinear"),
        eff_bit=draft_config.get("eff_bit", 0.1),
        split_dim=draft_config.get("split_dim", 1024),
        residual=draft_config.get("residual", False),
        kv_factor=draft_config.get("kv_factor", 1.0),
        min_split_dim=draft_config.get("min_split_dim", 8),
        model_id=args.draft_model_path,
    )
    
    draft_model = load_quantized_model(
        model_path=args.draft_model_path,
        quant_args=draft_args,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    # ===== Step C: Compute residual weights =====
    logger.info("Computing residual weights (W_original - W_draft_approx)...")
    residual_weights = compute_residual_weights(original_model, draft_model, device='cpu')
    
    # Free original model memory
    del original_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ===== Step D: Create and initialize residual model (0.9-bit) =====
    logger.info("Creating 0.9-bit residual model...")
    residual_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    residual_model.config.use_cache = False
    
    # Apply LittleBit patch with 0.9-bit configuration
    logger.info(f"Applying LittleBit patch with eff_bit={args.eff_bit}...")
    residual_model = apply_littlebit_patch(residual_model, args, do_train=True)
    
    # Re-initialize from residual weights
    logger.info("Initializing residual model from computed residual weights...")
    initialize_residual_model_weights(residual_model, residual_weights)
    del residual_weights
    gc.collect()
    
    # ===== Step E: Create combined Matryoshka model =====
    logger.info("Creating Matryoshka combined model (draft frozen + residual trainable)...")
    combined_model = MatryoshkaResidualModel(
        draft_model=draft_model,
        residual_model=residual_model,
        config=residual_model.config,
    )
    
    # Enable gradient for residual model, ensure input grads flow
    for name, param in combined_model.residual_model.named_parameters():
        if any(key in name for key in ("lm_head", "embed")):
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    if hasattr(combined_model.residual_model, "enable_input_require_grads"):
        combined_model.residual_model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        combined_model.residual_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    combined_model.residual_model.gradient_checkpointing_enable()
    
    if device_map:
        target = device_map if isinstance(device_map, (str, torch.device)) else list(device_map.values())[0]
        combined_model.to(target)
    
    print_trainable_parameters(combined_model.residual_model)
    
    # ===== Step F: Train =====
    logger.info("Loading teacher model for KD...")
    teacher_model = load_teacher_model(args, num_gpus, torch.bfloat16)
    
    training_args = get_training_arguments(args, save_dir)
    
    trainer = MatryoshkaKDTrainer(
        teacher_model=teacher_model,
        l2l_loss_scale=args.l2l_loss_scale,
        model=combined_model,
        processing_class=tokenizer,
        train_dataset=datasets,
        args=training_args,
        data_collator=default_data_collator,
    )
    
    logger.info("Starting Step 2 QAT training (0.9-bit residual model)...")
    trainer.train()
    
    # ===== Step G: Save =====
    save_residual_artifacts(trainer, combined_model, tokenizer, save_dir, args)
    
    logger.info("=" * 60)
    logger.info("Step 2 training complete!")
    logger.info(f"Residual model saved to: {save_dir}")
    logger.info(f"Draft model path: {args.draft_model_path}")
    logger.info("You can now run speculative_decoding.py with both models.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
