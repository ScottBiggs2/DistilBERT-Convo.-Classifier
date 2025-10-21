#!/usr/bin/env python3
"""
Fine-tune DistilBERT for conversation intent classification
"""

import json
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intent labels - must match your classification scheme
INTENT_LABELS = [
    "academic_help",
    "personal_writing_or_communication", 
    "writing_and_editing",
    "write_fiction",
    "how_to_advice",
    "creative_ideation",
    "translation",
    "computer_programming",
    "purchasable_products",
    "cooking_and_recipes",
    "health_fitness_beauty_or_self_care",
    "specific_info",
    "greetings_and_chitchat",
    "relationships_and_personal_reflection",
    "games_and_role_play",
    "media_generation_or_analysis", 
    "unclear",
    "other"
]

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    output_dir: str = "./models/distilbert-intent-classifier"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    skip_evaluation: bool = False  # Skip evaluation for faster training
    test_size: float = 0.15  # Held-out test set size
    val_size: float = 0.15   # Validation set size (from remaining data)

class ConversationDataset(Dataset):
    def __init__(self, conversations: List[str], labels: List[str], tokenizer, max_length: int):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label to id mapping
        self.label_to_id = {label: idx for idx, label in enumerate(INTENT_LABELS)}
        self.id_to_label = {idx: label for idx, label in enumerate(INTENT_LABELS)}
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = str(self.conversations[idx])
        label = self.labels[idx]
        
        # Tokenize conversation
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_to_id[label], dtype=torch.long)
        }

def load_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load conversation data from JSON file"""
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    labels = []
    
    for item in data:
        # Use the truncated sequence for training (cleaner)
        conversation = item.get('sequence_truncated', item.get('sequence', ''))
        intent = item.get('intent', 'unclear')
        
        if conversation and intent in INTENT_LABELS:
            conversations.append(conversation)
            labels.append(intent)
    
    logger.info(f"Loaded {len(conversations)} conversations with {len(set(labels))} unique labels")
    return conversations, labels

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def train_model(config: ModelConfig, data_path: str):
    """Main training function"""
    logger.info("Starting training pipeline...")
    
    # Load data
    conversations, labels = load_data(data_path)
    
    # Split data into train/val/test (only if not skipping evaluation)
    if config.skip_evaluation:
        train_texts, train_labels = conversations, labels
        val_texts, val_labels = [], []
        test_texts, test_labels = [], []
        logger.info(f"Training on full dataset: {len(train_texts)} samples (no validation or test)")
    else:
        # First split: separate test set (held-out)
        temp_texts, test_texts, temp_labels, test_labels = train_test_split(
            conversations, labels, 
            test_size=config.test_size, 
            random_state=42, 
            stratify=labels
        )
        
        # Second split: train/validation from remaining data
        val_size_adjusted = config.val_size / (1 - config.test_size)  # Adjust val size for remaining data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=temp_labels
        )
        
        logger.info(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(INTENT_LABELS),
        id2label={i: label for i, label in enumerate(INTENT_LABELS)},
        label2id={label: i for i, label in enumerate(INTENT_LABELS)}
    )
    
    # Create datasets
    train_dataset = ConversationDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = ConversationDataset(val_texts, val_labels, tokenizer, config.max_length) if not config.skip_evaluation else None
    test_dataset = ConversationDataset(test_texts, test_labels, tokenizer, config.max_length) if not config.skip_evaluation else None
    
    # Training arguments - conditional evaluation setup
    training_args_dict = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.num_epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "logging_dir": f"{config.output_dir}/logs",
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 2,
        "report_to": None,
        "dataloader_num_workers": 2,
        "learning_rate": config.learning_rate,
    }
    
    # Add evaluation settings only if not skipping
    if not config.skip_evaluation:
        training_args_dict.update({
            "eval_strategy": "steps",
            "eval_steps": config.eval_steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_f1_weighted",
            "greater_is_better": True,
        })
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Initialize trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
    }
    
    # Add evaluation components only if not skipping
    if not config.skip_evaluation:
        trainer_kwargs.update({
            "eval_dataset": val_dataset,
            "compute_metrics": compute_metrics,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)]
        })
    
    trainer = Trainer(**trainer_kwargs)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Final evaluation (only if not skipping)
    if not config.skip_evaluation:
        logger.info("Running validation evaluation...")
        val_results = trainer.evaluate(eval_dataset=val_dataset)
        
        logger.info("Running final test set evaluation...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        
        # Generate detailed classification report on TEST SET
        test_predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(test_predictions.predictions, axis=1)
        y_true = test_predictions.label_ids
        
        # Convert indices back to labels for reporting
        pred_labels = [INTENT_LABELS[i] for i in y_pred]
        true_labels = [INTENT_LABELS[i] for i in y_true]
        
        # Print detailed results
        print("\n" + "="*60)
        print("FINAL EVALUATION RESULTS")
        print("="*60)
        print("VALIDATION SET:")
        print(f"  Accuracy: {val_results['eval_accuracy']:.4f}")
        print(f"  F1 Macro: {val_results['eval_f1_macro']:.4f}")
        print(f"  F1 Weighted: {val_results['eval_f1_weighted']:.4f}")
        
        print("\nTEST SET (HELD-OUT):")
        print(f"  Accuracy: {test_results['eval_accuracy']:.4f}")
        print(f"  F1 Macro: {test_results['eval_f1_macro']:.4f}")
        print(f"  F1 Weighted: {test_results['eval_f1_weighted']:.4f}")
        
        print(f"\nTest Set Classification Report ({len(test_texts)} samples):")
        # Use labels parameter to handle missing classes in small datasets
        print(classification_report(true_labels, pred_labels, labels=INTENT_LABELS, target_names=INTENT_LABELS, zero_division=0))
        
        # Save evaluation results
        results_path = os.path.join(config.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'val_results': val_results,
                'test_results': test_results,
                'test_classification_report': classification_report(true_labels, pred_labels, labels=INTENT_LABELS, output_dict=True, zero_division=0),
                'data_splits': {
                    'train_size': len(train_texts),
                    'val_size': len(val_texts),
                    'test_size': len(test_texts)
                }
            }, f, indent=2)
        
        eval_results = test_results  # Return test results as main results
    else:
        logger.info("Skipping evaluation as requested")
        eval_results = {"message": "evaluation_skipped"}
    
    logger.info(f"Training complete! Model saved to {config.output_dir}")
    return trainer, eval_results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DistilBERT for intent classification")
    parser.add_argument("--data-path", default="./data/gpt4o_labelled.json", help="Path to labeled JSON data")
    parser.add_argument("--output-dir", default="./models/distilbert-intent-classifier", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation for faster training")
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        output_dir=args.output_dir,
        skip_evaluation=args.skip_eval
    )
    
    # Train model
    trainer, results = train_model(config, args.data_path)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {config.output_dir}")
    if not config.skip_evaluation and 'eval_f1_weighted' in results:
        print(f"Best F1 Score: {results['eval_f1_weighted']:.4f}")
    else:
        print("Training completed without evaluation")

if __name__ == "__main__":
    main()