#!/usr/bin/env python3
"""
Preprocess chat sequences:
- Split by [USER] / [ASSISTANT] markers
- Take last 5 turns
- Keep everything if ≤1024 tokens
- Otherwise, keep first 512 + last 512 tokens
- Lowercase everything
- Save as JSON, preserving all chat_ids
"""

import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

MAX_TOKENS = 1024
HALF_TOKENS = 512  # first + last 512 tokens if trimmed

def process_conversation_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []

    for convo in data:
        chat_id = convo.get("chat_id", "")
        text = convo.get("sequence", "")
        num_turns = convo.get("num_turns", None)

        if not text or not isinstance(text, str):
            # still preserve the chat_id with empty sequence
            processed.append({"chat_id": chat_id, "sequence": "", "num_turns": num_turns})
            continue

        text = text.lower()

        # Split into turns by [USER] / [ASSISTANT]
        segments = []
        for seg in text.split("[user]"):
            sub_segs = seg.split("[assistant]")
            for s in sub_segs:
                s = s.strip()
                if s:
                    segments.append(s)

        # Take last 5 turns (if fewer than 5, take all)
        last_turns = segments[-5:]

        # Rebuild sequence with markers
        sequence_text = ""
        for i, msg in enumerate(last_turns):
            role = "[USER]" if i % 2 == 0 else "[ASSISTANT]"
            sequence_text += f"{role} {msg} "

        # Tokenize
        tokens = tokenizer.encode(sequence_text, add_special_tokens=False)

        if len(tokens) <= MAX_TOKENS:
            trimmed_text = tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            # Keep first 512 + last 512 tokens
            first_half = tokens[:HALF_TOKENS]
            last_half = tokens[-HALF_TOKENS:]
            trimmed_text = tokenizer.decode(first_half, skip_special_tokens=True) + \
                           " [...] " + \
                           tokenizer.decode(last_half, skip_special_tokens=True)

        processed.append({
            "chat_id": chat_id,
            "sequence": trimmed_text,
            "num_turns": num_turns
        })

    # Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(processed)} sequences to {output_path}")

if __name__ == "__main__":
    input_file = "data/raw_sequences.json"
    output_file = "data/cleaned_sequences.json"
    process_conversation_file(input_file, output_file)
