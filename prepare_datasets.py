"""
Dataset Preparation Script for Matryoshka Speculative Decoding

Downloads and prepares:
1. Wikitext-2 (from HuggingFace)
2. ShareGPT (from HuggingFace: anon8231489123/ShareGPT_Vicuna_unfiltered)

Combines and tokenizes both into 2048-length chunks for QAT training.
Saves the processed dataset to disk for fast reuse.

Usage:
    python prepare_datasets.py \
        --model_id meta-llama/Llama-3.1-8B-Instruct \
        --output_dir ./data \
        --sharegpt_path /path/to/sharegpt.jsonl  # (optional, downloads from HF if not set)
"""

import argparse
import json
import os
import hashlib
import re
from itertools import chain

import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from utils.misc import setup_logger

logger = setup_logger(__name__)


def load_tokenizer(model_id):
    try:
        print(f"Loading Fast Tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except (OSError, TypeError, ValueError):
        print(f"Fast Tokenizer not found. Falling back to Slow Tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def download_wikitext2():
    """Download and return wikitext-2-raw-v1 train split."""
    logger.info("Downloading Wikitext-2 dataset from HuggingFace...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    num_chars = len(text)
    logger.info(f"  Wikitext-2 loaded: {len(dataset)} rows, {num_chars:,} characters")
    return text


def download_sharegpt(sharegpt_path=None):
    """Download/load ShareGPT data and return combined text."""
    sharegpt_texts = []

    if sharegpt_path and os.path.exists(sharegpt_path):
        logger.info(f"Loading ShareGPT from local file: {sharegpt_path}")
        with open(sharegpt_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "conversations" in item:
                        for turn in item["conversations"]:
                            content = turn.get("value", turn.get("content", ""))
                            if content:
                                sharegpt_texts.append(content)
                    elif "text" in item:
                        sharegpt_texts.append(item["text"])
                except json.JSONDecodeError:
                    continue
        logger.info(f"  Loaded {len(sharegpt_texts)} conversation turns from local file")
    else:
        logger.info("Downloading ShareGPT from HuggingFace (anon8231489123/ShareGPT_Vicuna_unfiltered)...")
        try:
            sharegpt_dataset = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                "default",
                split="train",
            )
            for item in sharegpt_dataset:
                if "conversations" in item and item["conversations"]:
                    for turn in item["conversations"]:
                        content = turn.get("value", turn.get("content", ""))
                        if content:
                            sharegpt_texts.append(content)
            logger.info(f"  ShareGPT loaded: {len(sharegpt_dataset)} conversations, {len(sharegpt_texts)} turns")
        except Exception as e:
            logger.warning(f"Failed to load ShareGPT from HuggingFace: {e}")
            logger.warning("Falling back to alternative ShareGPT source...")
            try:
                sharegpt_dataset = load_dataset(
                    "RyokoAI/ShareGPT52K",
                    split="train",
                )
                for item in sharegpt_dataset:
                    if "conversations" in item and item["conversations"]:
                        for turn in item["conversations"]:
                            content = turn.get("value", turn.get("content", ""))
                            if content:
                                sharegpt_texts.append(content)
                logger.info(f"  ShareGPT (alt) loaded: {len(sharegpt_texts)} turns")
            except Exception as e2:
                logger.error(f"All ShareGPT sources failed: {e2}")
                logger.error("Please provide a local ShareGPT file via --sharegpt_path")
                return ""

    combined = "\n\n".join(sharegpt_texts)
    logger.info(f"  ShareGPT total: {len(combined):,} characters")
    return combined


def tokenize_and_chunk(text, tokenizer, block_size=2048):
    """Tokenize text and split into fixed-length chunks."""
    logger.info(f"Tokenizing {len(text):,} characters with block_size={block_size}...")

    combined_dataset = datasets.Dataset.from_dict({"text": [text]})
    combined_dataset = (
        combined_dataset
        .add_column(name="timestamp", column=[None])
        .add_column(name="url", column=combined_dataset["text"])
    )

    column_names = list(combined_dataset.features)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = combined_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    processed = tokenized.map(group_texts, batched=True)
    logger.info(f"  Processed: {len(processed)} samples of {block_size} tokens each")
    return processed


def get_tokenizer_hash(tokenizer):
    """Compute a hash for the tokenizer for caching."""
    try:
        if tokenizer.is_fast:
            hash_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer.name_or_path, use_fast=False, trust_remote_code=True
            )
            text = hash_tokenizer.__repr__()
        else:
            text = tokenizer.__repr__()
    except Exception:
        text = tokenizer.__repr__()
        text = re.sub(r"TokenizerFast", "Tokenizer", text)

    hash_key = re.sub(r"name_or_path=[^,]+,?\s*", "", text)
    return hashlib.sha256(hash_key.encode()).hexdigest()[:7]


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Matryoshka QAT training")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model ID for tokenizer")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Root directory for saving processed datasets")
    parser.add_argument("--sharegpt_path", type=str, default=None,
                        help="Path to local ShareGPT .jsonl file (downloads from HF if not set)")
    parser.add_argument("--block_size", type=int, default=2048,
                        help="Token sequence length for training chunks")
    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = load_tokenizer(args.model_id)
    tok_hash = get_tokenizer_hash(tokenizer)

    # ============================
    # 1. Wikitext-2 only
    # ============================
    logger.info("\n" + "=" * 60)
    logger.info("Preparing: wikitext2")
    logger.info("=" * 60)
    wiki_text = download_wikitext2()
    wiki_dataset = tokenize_and_chunk(wiki_text, tokenizer, args.block_size)

    wiki_save_path = os.path.join(args.output_dir, "wikitext2", tok_hash)
    os.makedirs(wiki_save_path, exist_ok=True)
    wiki_dataset.save_to_disk(wiki_save_path)
    logger.info(f"  Saved to: {wiki_save_path}")

    # ============================
    # 2. ShareGPT
    # ============================
    logger.info("\n" + "=" * 60)
    logger.info("Preparing: ShareGPT")
    logger.info("=" * 60)
    sharegpt_text = download_sharegpt(args.sharegpt_path)

    if sharegpt_text:
        # Save raw ShareGPT for reuse
        raw_sharegpt_path = os.path.join(args.output_dir, "sharegpt_raw.txt")
        with open(raw_sharegpt_path, "w", encoding="utf-8") as f:
            f.write(sharegpt_text)
        logger.info(f"  Raw ShareGPT saved to: {raw_sharegpt_path}")

    # ============================
    # 3. Combined: wikitext2 + ShareGPT
    # ============================
    logger.info("\n" + "=" * 60)
    logger.info("Preparing: wikitext2_sharegpt (combined)")
    logger.info("=" * 60)

    combined_text = wiki_text
    if sharegpt_text:
        combined_text = wiki_text + "\n\n" + sharegpt_text
    else:
        logger.warning("ShareGPT was empty, using wikitext2 only!")

    combined_dataset = tokenize_and_chunk(combined_text, tokenizer, args.block_size)

    combined_save_path = os.path.join(args.output_dir, "wikitext2_sharegpt", tok_hash)
    os.makedirs(combined_save_path, exist_ok=True)
    combined_dataset.save_to_disk(combined_save_path)
    logger.info(f"  Saved to: {combined_save_path}")

    # ============================
    # Summary
    # ============================
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Tokenizer:      {args.model_id}")
    print(f"  Tokenizer hash: {tok_hash}")
    print(f"  Block size:     {args.block_size}")
    print()
    print(f"  wikitext2:          {wiki_save_path}  ({len(wiki_dataset)} samples)")
    print(f"  wikitext2_sharegpt: {combined_save_path}  ({len(combined_dataset)} samples)")
    print()
    print("  These datasets will be auto-detected by training scripts.")
    print("  Just set --data_root to match --output_dir above.")
    print(f"    --data_root {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
