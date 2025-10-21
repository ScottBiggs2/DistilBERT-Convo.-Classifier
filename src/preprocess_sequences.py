#!/usr/bin/env python3
"""
Preprocess conversation sequences:
- Load raw_sequences.json (one record per chat)
- Clean text (remove control chars, URLs)
- Batch tokenize and truncate at token-level using a front+back strategy
- Save cleaned/truncated sequences to data/cleaned_sequences.json

Outputs records:
{ "chat_id": "...", "sequence": "...", "sequence_truncated": "...", "num_turns": N }
"""

import os
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

RAW_SEQUENCES_PATH = os.getenv("RAW_SEQUENCES_PATH", "data/raw_sequences.json")
CLEANED_OUTPUT_PATH = os.getenv("CLEANED_SEQUENCES_PATH", "data/cleaned_sequences.json")

TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "distilbert-base-uncased")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
FRONT_RATIO = float(os.getenv("FRONT_TOKEN_RATIO", 0.75))  # proportion of tokens kept at front
BATCH_SIZE = int(os.getenv("PREPROCESS_BATCH_SIZE", 64))
USE_FAST_TOKENIZER = True  # use HuggingFace fast tokenizer

# load tokenizer (no model)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=USE_FAST_TOKENIZER)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)  # control chars
    return text.strip()

def truncate_tokens_front_back(token_ids, max_tokens=MAX_TOKENS, front_ratio=FRONT_RATIO):
    """
    Keep first floor(max_tokens*front_ratio) tokens and last (max_tokens - front) tokens.
    token_ids: list[int]
    """
    if len(token_ids) <= max_tokens:
        return token_ids
    front_len = int(max_tokens * front_ratio)
    back_len = max_tokens - front_len
    if front_len <= 0:
        front_len = max_tokens // 2
        back_len = max_tokens - front_len
    return token_ids[:front_len] + token_ids[-back_len:]

def batch_truncate_sequences(sequences):
    """
    sequences: list of raw sequence strings
    returns list of truncated strings
    """
    # tokenizer.batch_encode_plus returns dict with input_ids list
    enc = tokenizer.batch_encode_plus(
        sequences,
        add_special_tokens=True,
        truncation=False,  # we will truncate manually for head+tail behavior
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    truncated_texts = []
    for ids in enc["input_ids"]:
        t_ids = truncate_tokens_front_back(ids, MAX_TOKENS, FRONT_RATIO)
        text = tokenizer.decode(t_ids, skip_special_tokens=True)
        truncated_texts.append(text)
    return truncated_texts

def preprocess(input_path=RAW_SEQUENCES_PATH, output_path=CLEANED_OUTPUT_PATH):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    raw_sequences = []
    metas = []
    for it in items:
        seq = it.get("sequence") or it.get("message") or ""
        seq = clean_text(seq)
        if not seq:
            continue
        raw_sequences.append(seq)
        metas.append({
            "chat_id": it.get("chat_id"),
            "num_turns": it.get("num_turns", None),
            # keep any other metadata you want (but not "context"/"query")
        })

    truncated_results = []
    for i in tqdm(range(0, len(raw_sequences), BATCH_SIZE), desc="Truncate batches"):
        batch = raw_sequences[i:i+BATCH_SIZE]
        truncated = batch_truncate_sequences(batch)
        truncated_results.extend(truncated)

    processed = []
    for meta, orig, trunc in zip(metas, raw_sequences, truncated_results):
        processed.append({
            "chat_id": meta.get("chat_id"),
            "sequence": orig,
            "sequence_truncated": trunc,
            "num_turns": meta.get("num_turns")
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(processed, fout, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(processed)} cleaned sequences to {output_path}")

if __name__ == "__main__":
    preprocess()
