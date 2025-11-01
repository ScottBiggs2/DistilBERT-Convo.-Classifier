#!/usr/bin/env python3
"""
Preprocess data for evaluation.

- Streams records from data/records.json to handle large files.
- Extracts and cleans conversation sequences using logic from existing scripts.
- Takes a sample of the data for the evaluation set.
- Saves the result to eval/evaluation_dataset.json.
"""

import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import ijson

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Configuration
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "records.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "evaluation_dataset.json")
SAMPLE_SIZE = 1000
MAX_TURNS = 5
MAX_TOKENS = 1024
HALF_TOKENS = 512

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def build_sequence_from_chat(chat: dict, max_turns: int = MAX_TURNS):
    """
    Build a single string sequence from a chat's messages.
    """
    msgs = chat.get("messages", [])
    if not msgs:
        return None

    try:
        msgs_sorted = sorted(msgs, key=lambda m: m.get("createdAt", ""))
    except Exception:
        msgs_sorted = msgs

    msgs_slice = msgs_sorted[-max_turns:]

    parts = []
    for m in msgs_slice:
        role = m.get("role", "").lower()
        content = m.get("content", "")
        if not content or not isinstance(content, str):
            continue
        content = content.strip()
        if role == "user":
            parts.append("[USER] " + content)
        elif role == "assistant":
            parts.append("[ASSISTANT] " + content)
        else:
            parts.append("[OTHER] " + content)

    if not parts:
        return None

    sequence_text = " \n".join(parts)
    return {
        "chat_id": chat.get("chat_id"),
        "sequence": sequence_text,
        "num_turns": len(parts),
        "source": "records.json"
    }

def process_and_truncate_sequence(sequence: dict):
    """
    Process and truncate a single sequence.
    """
    text = sequence.get("sequence", "")
    if not text or not isinstance(text, str):
        return {**sequence, "text_for_model": ""}

    text = text.lower()

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= MAX_TOKENS:
        trimmed_text = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        first_half = tokens[:HALF_TOKENS]
        last_half = tokens[-HALF_TOKENS:]
        trimmed_text = tokenizer.decode(first_half, skip_special_tokens=True) + \
                       " [...] " + \
                       tokenizer.decode(last_half, skip_special_tokens=True)

    return {**sequence, "text_for_model": trimmed_text}

def create_evaluation_dataset():
    """Create the evaluation dataset from records.json."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    processed_sequences = []
    print(f"Loading all records from {INPUT_PATH} to sample from the end...")

    # Read all records into memory (assumes the file is a JSON array)
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        all_chats = json.load(f)

    # Take the last SAMPLE_SIZE records
    last_chats = all_chats[-SAMPLE_SIZE:]

    for chat in tqdm(last_chats, desc="Processing records (from end)"):
        raw_sequence = build_sequence_from_chat(chat)
        if raw_sequence:
            processed_sequence = process_and_truncate_sequence(raw_sequence)
            processed_sequences.append(processed_sequence)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        json.dump(processed_sequences, fout, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(processed_sequences)} processed sequences to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_evaluation_dataset()