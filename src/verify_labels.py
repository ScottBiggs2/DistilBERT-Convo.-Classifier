#!/usr/bin/env python3
"""
Compare and Filter Labels from Gemini and GPT-4o

This script compares the labels from two different teacher models (Gemini and GPT-4o),
calculates the agreement rate, generates a confusion matrix, and creates a new
distillation-ready JSON file containing only the samples where the models agree.
"""

import json
import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json(filepath: str):
    """Load a JSON or JSONL file."""
    logger.info(f"📂 Loading: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

def extract_emotion_from_raw_response(item):
    """Extract emotion from raw_response field, handling both 'intent' and 'emotion' keys"""
    raw = item.get("raw_response", "")
    
    # Try JSON parsing first
    try:
        parsed = json.loads(raw)
        # Try both 'emotion' and 'intent' keys (for backwards compatibility)
        emotion = parsed.get("emotion") or parsed.get("intent")
        if emotion and emotion in {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                                   '0', '1', '2'}:
            return emotion
    except json.JSONDecodeError:
        pass
    
    # Fallback: regex to find any single character A-Z, 0, 1, 2
    match = re.search(r'\b([A-Z]|[0-2])\b', raw)
    if match:
        return match.group(1)
    
    return None

def main():
    """Main function to compare and filter labels."""
    
    # File paths
    gemini_labels_path = "data/gemini_2.5_flash_labelled.json"
    gpt4o_labels_path = "data/distillation_data_gpt4o_mini.json"
    output_path = "data/agreed_distillation_data.json"
    confusion_matrix_path = "data/gemini_gpt_agreement_confusion_matrix.png"

    # Load the data
    gemini_data = load_json(gemini_labels_path)
    gpt4o_data = load_json(gpt4o_labels_path)

    # Extract and fix Gemini labels from raw_response
    logger.info("🔧 Extracting emotions from Gemini raw_response fields...")
    gemini_labels = {}
    extraction_stats = {'success': 0, 'failed': 0}
    
    for item in gemini_data:
        chat_id = item.get('chat_id')
        if not chat_id:
            continue
            
        # Extract emotion from raw_response
        extracted_emotion = extract_emotion_from_raw_response(item)
        
        if extracted_emotion:
            gemini_labels[chat_id] = extracted_emotion
            extraction_stats['success'] += 1
        else:
            extraction_stats['failed'] += 1
            logger.debug(f"Failed to extract emotion for chat_id {chat_id}: {item.get('raw_response', '')}")
    
    logger.info(f"✅ Successfully extracted {extraction_stats['success']} emotions")
    logger.info(f"❌ Failed to extract {extraction_stats['failed']} emotions")

    # Extract GPT-4o labels
    if isinstance(gpt4o_data, dict) and 'distillation_ready' in gpt4o_data:
        gpt4o_items = gpt4o_data['distillation_ready']
    else:
        gpt4o_items = gpt4o_data
    
    gpt4o_labels = {item['chat_id']: item['teacher_prediction'] for item in gpt4o_items}
    gpt4o_soft_labels = {item['chat_id']: item['soft_labels'] for item in gpt4o_items}
    gpt4o_text = {item['chat_id']: item['text'] for item in gpt4o_items}
    
    # Get class order from GPT-4o data
    if isinstance(gpt4o_data, dict) and 'metadata' in gpt4o_data:
        class_order = gpt4o_data['metadata']['class_order']
    else:
        # Default class order for 29 emotions
        class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      '0', '1', '2']

    # Debugging logs
    logger.info(f"Found {len(gemini_labels)} labels in {gemini_labels_path}")
    logger.info(f"Example chat_ids from Gemini: {list(gemini_labels.keys())[:5]}")

    logger.info(f"Found {len(gpt4o_labels)} labels in {gpt4o_labels_path}")
    logger.info(f"Example chat_ids from GPT-4o-mini: {list(gpt4o_labels.keys())[:5]}")

    # Show label distribution for Gemini
    gemini_label_dist = Counter(gemini_labels.values())
    logger.info(f"\n📊 Gemini Label Distribution:")
    for label, count in sorted(gemini_label_dist.items()):
        pct = count / len(gemini_labels) * 100 if gemini_labels else 0
        logger.info(f"   {label}: {count} ({pct:.1f}%)")

    # Show label distribution for GPT-4o-mini
    gpt4o_label_dist = Counter(gpt4o_labels.values())
    logger.info(f"\n📊 GPT-4o-mini Label Distribution:")
    for label, count in sorted(gpt4o_label_dist.items()):
        pct = count / len(gpt4o_labels) * 100 if gpt4o_labels else 0
        logger.info(f"   {label}: {count} ({pct:.1f}%)")

    # Find common chat_ids
    common_chat_ids = set(gemini_labels.keys()) & set(gpt4o_labels.keys())
    logger.info(f"\n🔗 Found {len(common_chat_ids)} common chat_ids between the two files.")

    # Compare labels and filter for agreement
    agreed_samples = []
    y_true = []
    y_pred = []
    agreement_count = 0

    for chat_id in common_chat_ids:
        gemini_label = gemini_labels[chat_id]
        gpt4o_label = gpt4o_labels[chat_id]
        
        y_true.append(gemini_label)
        y_pred.append(gpt4o_label)

        if gemini_label == gpt4o_label:
            agreement_count += 1
            agreed_samples.append({
                'chat_id': chat_id,
                'text': gpt4o_text[chat_id],
                'hard_label': gemini_label,  # or gpt4o_label, they are the same
                'teacher_prediction': gemini_label,
                'soft_labels': gpt4o_soft_labels[chat_id],
                'class_order': class_order
            })

    # Calculate and log agreement rate
    agreement_rate = agreement_count / len(common_chat_ids) if common_chat_ids else 0
    logger.info(f"\n✅ Agreement rate: {agreement_rate:.2%} ({agreement_count}/{len(common_chat_ids)})")

    # Generate and save confusion matrix
    if common_chat_ids:
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Gemini vs. GPT-4o-mini Label Agreement\nAgreement Rate: {agreement_rate:.2%}')
        plt.xlabel('GPT-4o-mini Label')
        plt.ylabel('Gemini Label')
        plt.tight_layout()
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Confusion matrix saved to {confusion_matrix_path}")
        plt.close()

    # Save the agreed samples to a new JSON file
    output_data = {
        "distillation_ready": agreed_samples,
        "metadata": {
            "class_order": class_order,
            "total_samples": len(agreed_samples),
            "agreement_rate": agreement_rate,
            "gemini_total": len(gemini_labels),
            "gpt4o_total": len(gpt4o_labels),
            "common_samples": len(common_chat_ids)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved {len(agreed_samples)} agreed samples to {output_path}")
    logger.info(f"🎯 Ready for knowledge distillation training!")

if __name__ == "__main__":
    main()