"""
Tree Attention Utilities for Speculative Decoding

Adapted from EAGLE (https://github.com/SafeAILab/EAGLE) for use with
Matryoshka LittleBit models.

Key concepts:
- tree_choices: list of paths, e.g. [[0],[1],[2],[0,0],[0,1],...]
  Each path represents a candidate sequence branching from the root.
  [0] = 1st top-k token at depth 1
  [0,1] = 1st token at depth 1, then 2nd token at depth 2
  
- tree_attn_mask: attention mask allowing each node to attend to its ancestors
- tree_position_ids: position offset for each node in the tree
- retrieve_indices: maps each leaf path back through the tree for verification
"""

import copy
import random
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F


# ==============================================================================
# Predefined Tree Structures
# ==============================================================================

# From EAGLE: mc_sim_7b_63 — 25 nodes, max depth 5, good balance of breadth/depth
TREE_CHOICES_DEFAULT = [
    [0], [1], [2], [3],
    [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0],
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1], [1, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
]

# Smaller tree for faster inference (fewer candidates)
TREE_CHOICES_SMALL = [
    [0], [1], [2],
    [0, 0], [0, 1], [1, 0],
    [0, 0, 0], [0, 0, 1],
    [0, 0, 0, 0],
]

# Larger tree for higher acceptance (more candidates)
TREE_CHOICES_LARGE = [
    [0], [1], [2], [3], [4],
    [0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0],
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 2, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1],
]

TREE_PRESETS = {
    "default": TREE_CHOICES_DEFAULT,
    "small": TREE_CHOICES_SMALL,
    "large": TREE_CHOICES_LARGE,
}


# ==============================================================================
# Tree Buffer Generation
# ==============================================================================

def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """Pad path list to desired length."""
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices: List[List[int]], device="cuda") -> Dict[str, torch.Tensor]:
    """
    Generate tree attention buffers from tree_choices structure.
    
    Args:
        tree_choices: List of paths defining the tree structure.
        device: Target device for tensors.
    
    Returns:
        Dictionary containing:
        - tree_attn_mask: (1, 1, tree_len, tree_len) attention mask
        - tree_position_ids: (tree_len,) position offsets for each node
        - retrieve_indices: (num_leaves, max_depth+1) indices to retrieve leaf paths
    """
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1  # +1 for root node

    # Count nodes at each depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Build tree attention mask
    # Each node can attend to itself and all its ancestors
    tree_attn_mask = torch.eye(tree_len, tree_len)
    tree_attn_mask[:, 0] = 1  # All nodes attend to root
    
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            if len(cur_tree_choice) == 1:
                continue
            # Find ancestor positions
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Build position IDs (depth of each node)
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Build retrieve indices for leaf-to-root paths
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        for c in range(len(cur_tree_choice)):
            retrieve_indice.append(
                sorted_tree_choices.index(cur_tree_choice[:c + 1])
            )
            retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    
    max_length = max(len(x) for x in retrieve_indices_nest)
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1  # offset for root
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
        dim=1,
    )

    # Sort retrieve indices for consistent ordering
    maxitem = retrieve_indices.max().item() + 5
    def custom_sort(lst):
        return [x if x >= 0 else maxitem for x in lst]
    
    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    # Aggregate buffers
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),  # (1,1,T,T)
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
        "sorted_tree_choices": sorted_tree_choices,
        "tree_len": tree_len,
        "depth_counts": depth_counts,
    }

    # Move to device
    tree_buffers = {
        k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
        for k, v in tree_buffers.items()
    }

    return tree_buffers


# ==============================================================================
# Posterior Evaluation — finding the best candidate path
# ==============================================================================

def evaluate_posterior_greedy(
    logits: torch.Tensor,
    candidates: torch.Tensor,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Greedy posterior evaluation: find the longest-matching candidate path.
    
    Args:
        logits: (num_candidates, seq_len, vocab_size) target model logits
                for each candidate path
        candidates: (num_candidates, seq_len) token IDs of candidate paths
    
    Returns:
        best_candidate: index of best candidate
        accept_length: number of accepted tokens (0 = only bonus token)
        bonus_logits: logits to sample the next token from
    """
    # Check if each candidate token matches the argmax of target logits
    # logits[:, i] predicts position i+1, candidates[:, i+1] is the token at i+1
    posterior_mask = (
        candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
    ).int()
    
    # Cumulative product: find longest prefix match
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    
    if accept_length == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    
    return best_candidate, accept_length.item(), logits[best_candidate, accept_length]


def evaluate_posterior_sampling(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Sampling-based posterior evaluation with rejection sampling.
    
    Args:
        logits: (num_candidates, seq_len, vocab_size)
        candidates: (num_candidates, seq_len)
        temperature: sampling temperature
    
    Returns:
        best_candidate: index of best candidate
        accept_length: number of accepted tokens
        sample_p: probability distribution for the next token
    """
    accept_length = 1
    accept_cand = candidates[0][:1]
    best_candidate = 0
    
    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break
        
        adjustflag = False
        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
        
        gt_logits = logits[fi, i - 1][None]
        if temperature > 1e-5:
            gt_logits = gt_logits / temperature
        gtp = torch.softmax(gt_logits[0], dim=0)
        
        candidates_set = []
        for j in range(candidates.shape[0]):
            if is_eq[j]:
                x = candidates[j, i]
                xi = x.item()
                if xi in candidates_set or xi == -1:
                    continue
                candidates_set.append(xi)
                r = random.random()
                px = gtp[xi]
                qx = 1.0  # uniform draft assumption
                acp = px / qx
                if r <= acp:
                    accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                    accept_length += 1
                    best_candidate = j
                    break
                else:
                    gtp[xi] = 0
                    gtp = gtp / (gtp.sum() + 1e-10)
                    adjustflag = True
    
    if adjustflag and accept_length != candidates.shape[1]:
        sample_p = gtp
    else:
        gt_logits = logits[best_candidate, accept_length - 1][None]
        if temperature > 1e-5:
            gt_logits = gt_logits / temperature
        sample_p = torch.softmax(gt_logits[0], dim=0)
    
    return torch.tensor(best_candidate), accept_length - 1, sample_p


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    greedy: bool = True,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Evaluate posterior and select best candidate path.
    
    Args:
        logits: (num_candidates, seq_len, vocab_size)
        candidates: (num_candidates, seq_len)
        greedy: whether to use greedy decoding
        temperature: sampling temperature
    
    Returns:
        best_candidate, accept_length, next_token_logits/probs
    """
    if greedy:
        return evaluate_posterior_greedy(logits, candidates)
    else:
        return evaluate_posterior_sampling(logits, candidates, temperature)


# ==============================================================================
# Tree Candidate Generation from Draft Model
# ==============================================================================

@torch.no_grad()
def generate_draft_tree(
    draft_model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tree_buffers: Dict,
    top_k: int = 10,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate tree-structured draft candidates using the draft model.
    
    Instead of generating K tokens sequentially (serial mode),
    generates a tree of candidates by taking top-K at each depth level.
    
    Args:
        draft_model: The draft model (MatryoshkaDraftModel)
        input_ids: (1, seq_len) prompt tokens
        attention_mask: (1, seq_len) attention mask
        tree_buffers: from generate_tree_buffers()
        top_k: number of top candidates at each expansion
        temperature: sampling temperature
    
    Returns:
        tree_candidates: (1, tree_len) token IDs for the tree
        tree_logits: (1, tree_len, vocab) logits for each tree node
    """
    sorted_tree_choices = tree_buffers["sorted_tree_choices"]
    tree_len = tree_buffers["tree_len"]
    depth_counts = tree_buffers["depth_counts"]
    device = input_ids.device
    
    # Step 1: Get logits for the last position (root prediction)
    outputs = draft_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    root_logits = outputs.logits[:, -1, :]  # (1, vocab)
    past_kv = outputs.past_key_values

    # Get top-K tokens from root
    if temperature > 1e-5:
        scaled_logits = root_logits / temperature
    else:
        scaled_logits = root_logits
    
    root_probs = F.log_softmax(scaled_logits, dim=-1)
    top_vals, top_ids = torch.topk(root_probs, top_k, dim=-1)  # (1, top_k)
    
    # Build mapping: sorted_tree_choices index → token ID
    # We need to expand the tree level by level
    tree_tokens = torch.zeros(tree_len, dtype=torch.long, device=device)
    tree_tokens[0] = input_ids[0, -1]  # root = last input token
    
    # Track logits and scores for each node
    all_logits = torch.zeros(tree_len, root_logits.shape[-1], device=device, dtype=root_logits.dtype)
    all_logits[0] = root_logits[0]
    
    # Map path → score (cumulative log prob)
    node_scores = {}  # path tuple → score
    node_tokens = {}  # path tuple → token id
    node_logits = {}  # path tuple → logits
    
    for k in range(top_k):
        path = (k,)
        node_scores[path] = top_vals[0, k].item()
        node_tokens[path] = top_ids[0, k].item()
    
    # For each node in sorted_tree_choices, we need to generate via the draft model
    # Group by depth for batch processing
    start = 0
    current_past_kv = past_kv
    
    for depth_idx in range(len(depth_counts)):
        depth_nodes = sorted_tree_choices[start:start + depth_counts[depth_idx]]
        
        # Collect tokens for this depth that we need to expand further
        # Check if any deeper nodes depend on these
        has_children = set()
        for choice in sorted_tree_choices:
            if len(choice) > depth_idx + 1:
                parent = tuple(choice[:depth_idx + 1])
                has_children.add(parent)
        
        # For depth 0 nodes: use top-K tokens from root
        if depth_idx == 0:
            for j, node in enumerate(depth_nodes):
                node_idx = start + j + 1  # +1 for root
                path = tuple(node)
                token_id = node_tokens.get(path)
                if token_id is not None:
                    tree_tokens[node_idx] = token_id
                    all_logits[node_idx] = root_logits[0]  # logits that produced this token
        else:
            # For deeper nodes: tokens already determined by parent expansion
            for j, node in enumerate(depth_nodes):
                node_idx = start + j + 1
                path = tuple(node)
                token_id = node_tokens.get(path)
                if token_id is not None:
                    tree_tokens[node_idx] = token_id
        
        # Now expand nodes that have children
        nodes_to_expand = []
        for node in depth_nodes:
            path = tuple(node)
            if path in has_children and path in node_tokens:
                nodes_to_expand.append((path, node_tokens[path]))
        
        if nodes_to_expand:
            # Run draft model on each node to get next-level logits
            for path, token_id in nodes_to_expand:
                expand_ids = torch.tensor([[token_id]], device=device, dtype=torch.long)
                
                # Build attention mask for this expansion
                if attention_mask is not None:
                    expand_mask = torch.cat([
                        attention_mask,
                        torch.ones(1, input_ids.shape[1] + depth_idx + 1 - attention_mask.shape[1],
                                   device=device, dtype=attention_mask.dtype)
                    ], dim=1) if attention_mask.shape[1] < input_ids.shape[1] + depth_idx + 1 else attention_mask
                else:
                    expand_mask = None
                
                # Simple forward (no KV cache reuse for tree — we do independent expansions)
                # Build the full prefix for this path
                prefix_tokens = [input_ids[0].tolist()]
                for p in path:
                    # We need the actual token for this path position
                    pass
                
                # For simplicity, do a full forward with the prefix + path tokens
                path_token_ids = []
                for d in range(len(path)):
                    sub_path = tuple(list(path)[:d + 1])
                    if sub_path in node_tokens:
                        path_token_ids.append(node_tokens[sub_path])
                
                full_ids = torch.cat([
                    input_ids,
                    torch.tensor([path_token_ids], device=device, dtype=torch.long)
                ], dim=1)
                
                full_mask = torch.ones_like(full_ids) if attention_mask is not None else None
                
                expand_out = draft_model.forward(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=False,
                )
                expand_logits = expand_out.logits[:, -1, :]  # (1, vocab)
                
                if temperature > 1e-5:
                    scaled = expand_logits / temperature
                else:
                    scaled = expand_logits
                
                expand_probs = F.log_softmax(scaled, dim=-1)
                child_vals, child_ids = torch.topk(expand_probs, top_k, dim=-1)
                
                parent_score = node_scores.get(path, 0.0)
                
                # Register children
                for k in range(top_k):
                    child_path = path + (k,)
                    node_scores[child_path] = parent_score + child_vals[0, k].item()
                    node_tokens[child_path] = child_ids[0, k].item()
                    node_logits[child_path] = expand_logits[0]
        
        start += depth_counts[depth_idx]
    
    # Fill in any remaining tokens from node_tokens
    for j, node in enumerate(sorted_tree_choices):
        node_idx = j + 1
        path = tuple(node)
        if path in node_tokens:
            tree_tokens[node_idx] = node_tokens[path]
    
    tree_candidates = tree_tokens.unsqueeze(0)  # (1, tree_len)
    
    return tree_candidates


# ==============================================================================
# Tree Attention Mask for Target Model
# ==============================================================================

def build_tree_attention_mask(
    prefix_len: int,
    tree_attn_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Build full attention mask for target model verification.
    
    The mask allows:
    - All positions to attend to the prefix (causal)
    - Tree nodes to attend to their ancestors (tree structure)
    
    Args:
        prefix_len: length of the prompt/prefix
        tree_attn_mask: (1, 1, tree_len, tree_len) from tree buffers
        device: target device
        dtype: attention mask dtype
    
    Returns:
        full_mask: (1, 1, prefix_len + tree_len, prefix_len + tree_len)
    """
    tree_len = tree_attn_mask.shape[-1]
    total_len = prefix_len + tree_len
    
    # Start with all-zero mask (attend everywhere in prefix)
    # Then use tree mask for the tree part
    full_mask = torch.zeros(1, 1, total_len, total_len, device=device, dtype=dtype)
    
    # Causal mask for prefix
    causal = torch.triu(
        torch.ones(prefix_len, prefix_len, device=device, dtype=dtype) * torch.finfo(dtype).min,
        diagonal=1,
    )
    full_mask[0, 0, :prefix_len, :prefix_len] = causal
    
    # Tree nodes attend to all prefix positions
    # full_mask[:, :, prefix_len:, :prefix_len] = 0  # already 0
    
    # Tree nodes attend to tree nodes according to tree_attn_mask
    tree_mask_bool = tree_attn_mask[0, 0]  # (tree_len, tree_len)
    tree_block = torch.where(
        tree_mask_bool.bool(),
        torch.zeros(tree_len, tree_len, device=device, dtype=dtype),
        torch.ones(tree_len, tree_len, device=device, dtype=dtype) * torch.finfo(dtype).min,
    )
    full_mask[0, 0, prefix_len:, prefix_len:] = tree_block
    
    # Prefix cannot attend to tree nodes
    full_mask[0, 0, :prefix_len, prefix_len:] = torch.finfo(dtype).min
    
    return full_mask
