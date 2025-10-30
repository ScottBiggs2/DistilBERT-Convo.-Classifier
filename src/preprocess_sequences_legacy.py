#!/usr/bin/env python3
"""
Preprocess conversation sequences with adaptive chunk selection:
- Clean text
- Split into chunks (~128 tokens)
- Rank chunks by TF-IDF informativeness
- Keep head, tail, and top-scoring middle chunks until max token budget
- Save cleaned/truncated sequences to data/cleaned_sequences.json
"""

import os
import json
import re
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

RAW_SEQUENCES_PATH = os.getenv("RAW_SEQUENCES_PATH", "data/raw_sequences.json")
CLEANED_OUTPUT_PATH = os.getenv("CLEANED_SEQUENCES_PATH", "data/cleaned_sequences.json")

TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "distilbert-base-uncased")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
BATCH_SIZE = int(os.getenv("PREPROCESS_BATCH_SIZE", 64))
USE_FAST_TOKENIZER = True

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=USE_FAST_TOKENIZER)

# ---------------- Cleaning -----------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()

# ---------------- Chunking -----------------
def split_into_chunks(text, chunk_size_tokens=128):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size_tokens):
        sub = tokens[i:i + chunk_size_tokens]
        chunks.append(tokenizer.decode(sub))
    return chunks

# ---------------- Scoring ------------------
def select_informative_chunks(text, max_tokens=MAX_TOKENS):
    chunks = split_into_chunks(text, chunk_size_tokens=max_tokens // 4)
    if len(chunks) <= 2:
        return text  # small text, no truncation

    # compute TF-IDF scores per chunk
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )
    tfidf = vectorizer.fit_transform(chunks)
    scores = np.asarray(tfidf.sum(axis=1)).ravel()

    # always keep head and tail
    head, tail = 0, len(chunks) - 1
    remaining = list(range(1, tail))
    # rank middle chunks by score
    ranked = sorted(remaining, key=lambda i: scores[i], reverse=True)

    # build combined text under token budget
    selected = [head] + ranked + [tail]
    combined_tokens = []
    for idx in selected:
        chunk_toks = tokenizer.encode(chunks[idx], add_special_tokens=False)
        if len(combined_tokens) + len(chunk_toks) > max_tokens:
            break
        combined_tokens.extend(chunk_toks)
    return tokenizer.decode(combined_tokens, skip_special_tokens=True)

# ---------------- Batch wrapper ------------
def batch_truncate_sequences(sequences):
    truncated_texts = []
    for text in tqdm(sequences, desc="Adaptive truncation"):
        truncated_texts.append(select_informative_chunks(text, MAX_TOKENS))
    return truncated_texts

# ---------------- Main ---------------------
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
        })
    
    # Cast sequences into lowercase
    for i in range(len(raw_sequences)):
        raw_sequences[i] = raw_sequences[i].lower()


    truncated_results = batch_truncate_sequences(raw_sequences)

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
