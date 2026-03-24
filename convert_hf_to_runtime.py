"""
Convert HF LittleBit checkpoint to CPU runtime or canonical format.

Formats:
  runtime:    u_sign_rt/v_sign_rt keys (for CPU Python kernel)
  canonical:  u_sign_packed/v_sign_packed keys (for GGUF conversion → llama.cpp)

Usage:
    python convert_hf_to_runtime.py \
        --input_path outputs/step1_draft_0.1bit/<timestamp> \
        --output_path outputs/step1_draft_0.1bit/<timestamp>_runtime

    python convert_hf_to_runtime.py \
        --input_path outputs/step1_draft_0.1bit/<timestamp> \
        --output_path outputs/step1_draft_0.1bit/<timestamp>_canonical \
        --format canonical
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_state_dict(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert HF LittleBit state dict keys to CPU runtime format."""
    runtime_state = {}

    # Group keys by LittleBitLinear module prefix
    # Find all unique prefixes that have U_packed
    prefixes = set()
    for key in hf_state:
        if key.endswith(".U_packed"):
            prefixes.add(key[: -len(".U_packed")])
        elif key.endswith(".U_R_packed"):
            prefixes.add(key[: -len(".U_R_packed")])

    converted_keys = set()

    for prefix in sorted(prefixes):
        # === Main branch ===
        if f"{prefix}.U_packed" in hf_state:
            runtime_state[f"{prefix}.u_sign_rt"] = hf_state[f"{prefix}.U_packed"]
            runtime_state[f"{prefix}.u_sign_rt_shape"] = hf_state[f"{prefix}.U_shape"]
            runtime_state[f"{prefix}.v_sign_rt"] = hf_state[f"{prefix}.V_packed"]
            runtime_state[f"{prefix}.v_sign_rt_shape"] = hf_state[f"{prefix}.V_shape"]
            runtime_state[f"{prefix}.u1"] = hf_state[f"{prefix}.u1"]
            runtime_state[f"{prefix}.v2"] = hf_state[f"{prefix}.v2"]

            # Fuse: mid = v1 * u2
            v1 = hf_state[f"{prefix}.v1"].to(torch.float32)
            u2 = hf_state[f"{prefix}.u2"].to(torch.float32)
            runtime_state[f"{prefix}.mid"] = (v1 * u2).contiguous()

            converted_keys.update([
                f"{prefix}.U_packed", f"{prefix}.U_shape",
                f"{prefix}.V_packed", f"{prefix}.V_shape",
                f"{prefix}.u1", f"{prefix}.u2",
                f"{prefix}.v1", f"{prefix}.v2",
            ])

        # === Residual branch ===
        if f"{prefix}.U_R_packed" in hf_state:
            runtime_state[f"{prefix}.u_sign_r_rt"] = hf_state[f"{prefix}.U_R_packed"]
            runtime_state[f"{prefix}.u_sign_r_rt_shape"] = hf_state[f"{prefix}.U_R_shape"]
            runtime_state[f"{prefix}.v_sign_r_rt"] = hf_state[f"{prefix}.V_R_packed"]
            runtime_state[f"{prefix}.v_sign_r_rt_shape"] = hf_state[f"{prefix}.V_R_shape"]
            runtime_state[f"{prefix}.u1_r"] = hf_state[f"{prefix}.u1_R"]
            runtime_state[f"{prefix}.v2_r"] = hf_state[f"{prefix}.v2_R"]

            v1_r = hf_state[f"{prefix}.v1_R"].to(torch.float32)
            u2_r = hf_state[f"{prefix}.u2_R"].to(torch.float32)
            runtime_state[f"{prefix}.mid_r"] = (v1_r * u2_r).contiguous()

            converted_keys.update([
                f"{prefix}.U_R_packed", f"{prefix}.U_R_shape",
                f"{prefix}.V_R_packed", f"{prefix}.V_R_shape",
                f"{prefix}.u1_R", f"{prefix}.u2_R",
                f"{prefix}.v1_R", f"{prefix}.v2_R",
            ])

    # Pass through non-quantized tensors (layernorm, embed_tokens, buffers, etc.)
    for key, value in hf_state.items():
        if key not in converted_keys:
            # Skip internal LittleBit buffers
            if any(key.endswith(s) for s in [
                "._eff_bit_target", "._split_dim_final", "._eff_bit_actual",
                "._binarized",
            ]):
                continue
            runtime_state[key] = value

    return runtime_state


def convert_state_dict_canonical(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert HF LittleBit state dict to canonical format (for GGUF conversion).
    
    Key mapping:
        U_packed → u_sign_packed,  U_shape → u_sign_shape
        V_packed → v_sign_packed,  V_shape → v_sign_shape
        v1 * u2  → mid
        u1, v2   → same
    """
    canonical_state = {}
    prefixes = set()
    for key in hf_state:
        if key.endswith(".U_packed"):
            prefixes.add(key[: -len(".U_packed")])
        elif key.endswith(".U_R_packed"):
            prefixes.add(key[: -len(".U_R_packed")])

    converted_keys = set()

    for prefix in sorted(prefixes):
        if f"{prefix}.U_packed" in hf_state:
            canonical_state[f"{prefix}.u_sign_packed"] = hf_state[f"{prefix}.U_packed"]
            canonical_state[f"{prefix}.u_sign_shape"] = hf_state[f"{prefix}.U_shape"]
            canonical_state[f"{prefix}.v_sign_packed"] = hf_state[f"{prefix}.V_packed"]
            canonical_state[f"{prefix}.v_sign_shape"] = hf_state[f"{prefix}.V_shape"]
            canonical_state[f"{prefix}.u1"] = hf_state[f"{prefix}.u1"]
            canonical_state[f"{prefix}.v2"] = hf_state[f"{prefix}.v2"]

            v1 = hf_state[f"{prefix}.v1"].to(torch.float32)
            u2 = hf_state[f"{prefix}.u2"].to(torch.float32)
            canonical_state[f"{prefix}.mid"] = (v1 * u2).contiguous()

            converted_keys.update([
                f"{prefix}.U_packed", f"{prefix}.U_shape",
                f"{prefix}.V_packed", f"{prefix}.V_shape",
                f"{prefix}.u1", f"{prefix}.u2",
                f"{prefix}.v1", f"{prefix}.v2",
            ])

        if f"{prefix}.U_R_packed" in hf_state:
            canonical_state[f"{prefix}.u_sign_r_packed"] = hf_state[f"{prefix}.U_R_packed"]
            canonical_state[f"{prefix}.u_sign_r_shape"] = hf_state[f"{prefix}.U_R_shape"]
            canonical_state[f"{prefix}.v_sign_r_packed"] = hf_state[f"{prefix}.V_R_packed"]
            canonical_state[f"{prefix}.v_sign_r_shape"] = hf_state[f"{prefix}.V_R_shape"]
            canonical_state[f"{prefix}.u1_r"] = hf_state[f"{prefix}.u1_R"]
            canonical_state[f"{prefix}.v2_r"] = hf_state[f"{prefix}.v2_R"]

            v1_r = hf_state[f"{prefix}.v1_R"].to(torch.float32)
            u2_r = hf_state[f"{prefix}.u2_R"].to(torch.float32)
            canonical_state[f"{prefix}.mid_r"] = (v1_r * u2_r).contiguous()

            converted_keys.update([
                f"{prefix}.U_R_packed", f"{prefix}.U_R_shape",
                f"{prefix}.V_R_packed", f"{prefix}.V_R_shape",
                f"{prefix}.u1_R", f"{prefix}.u2_R",
                f"{prefix}.v1_R", f"{prefix}.v2_R",
            ])

    for key, value in hf_state.items():
        if key not in converted_keys:
            if any(key.endswith(s) for s in [
                "._eff_bit_target", "._split_dim_final", "._eff_bit_actual",
                "._binarized",
            ]):
                continue
            canonical_state[key] = value

    return canonical_state


def load_hf_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load state dict from HF-style safetensors checkpoint."""
    model_path = Path(model_path)
    state_dict = {}

    index_path = model_path / "model.safetensors.index.json"
    single_path = model_path / "model.safetensors"

    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        for shard_file in shard_files:
            shard_state = load_file(str(model_path / shard_file))
            state_dict.update(shard_state)
    elif single_path.exists():
        state_dict = load_file(str(single_path))
    else:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    return state_dict


def build_dummy_config(hf_config_path: str, littlebit_config_path: str) -> dict:
    """Build dummy_llama3_config.json from HF config.json and littlebit_config.json."""
    with open(hf_config_path, "r") as f:
        hf_config = json.load(f)
    with open(littlebit_config_path, "r") as f:
        lb_config = json.load(f)

    return {
        "name": hf_config.get("_name_or_path", "llama"),
        "hidden_size": hf_config["hidden_size"],
        "intermediate_size": hf_config["intermediate_size"],
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        "vocab_size": hf_config["vocab_size"],
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "include_lm_head": False,
        # Extra info
        "eff_bit": lb_config.get("eff_bit", 0.1),
        "residual": lb_config.get("residual", False),
    }


def build_runtime_config(hf_config_path: str, littlebit_config_path: str) -> dict:
    """Build littlebit_runtime_config.json."""
    with open(hf_config_path, "r") as f:
        hf_config = json.load(f)
    with open(littlebit_config_path, "r") as f:
        lb_config = json.load(f)

    return {
        "format_version": 1,
        "checkpoint_format": "runtime",
        "sign_storage": "row_i1",
        "runtime_layout": "cpu_row_i1",
        "base_model_id": hf_config.get("_name_or_path", None),
        "model_type": hf_config.get("model_type", "llama"),
        "torch_dtype": "bfloat16",
        "quant_func": lb_config.get("quant_func", "STEBinary"),
        "eff_bit": lb_config.get("eff_bit", 0.1),
        "split_dim": lb_config.get("split_dim", None),
        "residual": lb_config.get("residual", False),
    }


def convert_hf_to_runtime(input_path: str, output_path: str, fmt: str = "runtime"):
    """Convert HF LittleBit checkpoint to runtime or canonical format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF checkpoint from {input_path}...")
    hf_state = load_hf_state_dict(str(input_path))
    print(f"  Loaded {len(hf_state)} tensors")

    print(f"Converting to {fmt} format...")
    if fmt == "canonical":
        converted_state = convert_state_dict_canonical(hf_state)
        marker_suffix = ".u_sign_packed"
    else:
        converted_state = convert_state_dict(hf_state)
        marker_suffix = ".u_sign_rt"
    print(f"  Converted to {len(converted_state)} tensors")

    rt_keys = [k for k in converted_state if k.endswith(marker_suffix)]
    print(f"  Converted {len(rt_keys)} LittleBit linear projections")

    # For canonical format, save as model.safetensors (GGUF converter expects this)
    if fmt == "canonical":
        safetensors_path = output_path / "model.safetensors"
    else:
        safetensors_path = output_path / "littlebit_runtime.safetensors"
    print(f"Saving to {safetensors_path}...")
    save_file(converted_state, str(safetensors_path))

    # Save/copy configs
    hf_config_path = input_path / "config.json"
    lb_config_path = input_path / "littlebit_config.json"

    if hf_config_path.exists() and lb_config_path.exists():
        rt_config = build_runtime_config(str(hf_config_path), str(lb_config_path))
        rt_config_path = output_path / "littlebit_runtime_config.json"
        with open(rt_config_path, "w") as f:
            json.dump(rt_config, f, indent=2)

        dummy_config = build_dummy_config(str(hf_config_path), str(lb_config_path))
        dummy_config_path = output_path / "dummy_llama3_config.json"
        with open(dummy_config_path, "w") as f:
            json.dump(dummy_config, f, indent=2)

        # Copy config.json for GGUF converter
        if fmt == "canonical":
            import shutil
            shutil.copy2(str(hf_config_path), str(output_path / "config.json"))
            if lb_config_path.exists():
                shutil.copy2(str(lb_config_path), str(output_path / "littlebit_config.json"))
            # Copy tokenizer files
            for tok_file in input_path.glob("tokenizer*"):
                shutil.copy2(str(tok_file), str(output_path / tok_file.name))
            for tok_file in input_path.glob("special_tokens*"):
                shutil.copy2(str(tok_file), str(output_path / tok_file.name))
            print(f"Copied config and tokenizer files")

        print(f"Saved configs")

    del hf_state, converted_state
    print("Conversion complete!")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert HF LittleBit checkpoint to runtime/canonical format")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to HF LittleBit checkpoint directory")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for converted checkpoint")
    parser.add_argument("--format", type=str, default="runtime",
                        choices=["runtime", "canonical"],
                        help="Output format: 'runtime' (CPU Python kernel) or 'canonical' (GGUF conversion)")
    args = parser.parse_args()

    convert_hf_to_runtime(args.input_path, args.output_path, fmt=args.format)


if __name__ == "__main__":
    main()
