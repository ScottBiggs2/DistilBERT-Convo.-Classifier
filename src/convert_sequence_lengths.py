#!/usr/bin/env python3
"""
Preprocess chat sequences:
- Split by [USER] / [ASSISTANT] markers
- Take last 3 turns (6 messages)
- Concatenate and truncate the entire sequence to 512 tokens
- Lowercase everything
- Save as JSON, preserving all fields (chat_id, hard_label, soft_labels, class_order)
"""
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
MAX_TOKENS_TOTAL = 512

def process_conversation(data):
    processed = []
    for convo in data:
        chat_id = convo.get("chat_id", "")
        text = convo.get("text", "")
        hard_label = convo.get("hard_label", "")
        soft_labels = convo.get("soft_labels", [])
        class_order = convo.get("class_order", [])

        if not text or not isinstance(text, str):
            processed.append({
                "chat_id": chat_id,
                "text": "",
                "hard_label": hard_label,
                "soft_labels": soft_labels,
                "class_order": class_order
            })
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

        # Take last 3 turns (6 messages)
        last_turns = segments[-6:] if len(segments) >= 6 else segments
        # Rebuild sequence with markers
        sequence_text = ""
        for i, msg in enumerate(last_turns):
            role = "[USER]" if i % 2 == 0 else "[ASSISTANT]"
            sequence_text += f"{role} {msg} "

        # Truncate the entire sequence to 512 tokens
        tokens = tokenizer.encode(sequence_text, add_special_tokens=False)
        if len(tokens) > MAX_TOKENS_TOTAL:
            tokens = tokens[:MAX_TOKENS_TOTAL]
            sequence_text = tokenizer.decode(tokens, skip_special_tokens=True) + " [...]"

        processed.append({
            "chat_id": chat_id,
            "text": sequence_text,
            "hard_label": hard_label,
            "soft_labels": soft_labels,
            "class_order": class_order
        })

    return processed

def main(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Access the list of conversations under the "distillation_ready" key
    if "distillation_ready" in data:
        data = data["distillation_ready"]
    else:
        raise ValueError("Expected a 'distillation_ready' key in the input JSON.")

    processed = process_conversation(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(processed)} sequences to {output_file}")

if __name__ == "__main__":
    input_file = "data/agreed_distillation_data.json"  # Replace with your input file path
    output_file = "data/agreed_distillation_data_truncated.json"  # Replace with your output file path
    main(input_file, output_file)
