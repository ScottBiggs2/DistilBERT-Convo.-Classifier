#!/usr/bin/env python3
"""
DistilBERT Knowledge Distillation Training Script - Extended to 1024 Tokens

Features:
- Extended position embeddings to support 1024 tokens
- Combined hard label (CE) + soft label (KL divergence) losses
- Custom confusion-pair weighting (harsh NSFW/SFW penalties)
- Transformers-based training with easy ONNX export
- Comprehensive evaluation and confusion matrix analysis
- Optimized for conversation classification with cost-sensitive misclassification penalties
"""

import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import wandb

from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training"""
    
    # Model configuration
    model_name: str = "distilbert/distilbert-base-multilingual-cased"
    max_length: int = 1024  # Extended from 512
    num_labels: int = 13
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 1e-3
    
    # Distillation configuration
    temperature: float = 4.0
    alpha: float = 0.7
    
    # Loss weighting configuration
    banned_unbanned_penalty: float = 5.0
    within_category_penalty: float = 1.0
    
    # Evaluation configuration
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 500
    early_stopping_patience: int = 3


def extend_position_embeddings(model, new_max_length=1024):
    """
    Extend DistilBERT's position embeddings from 512 to new_max_length
    
    Strategy: Linear interpolation of existing embeddings to create smooth transitions
    This allows the model to handle longer sequences while preserving learned patterns.
    
    Args:
        model: DistilBertForSequenceClassification model
        new_max_length: Target maximum sequence length (default 1024)
    
    Returns:
        model: Modified model with extended position embeddings
    """
    old_embeddings = model.distilbert.embeddings.position_embeddings
    old_max_length = old_embeddings.weight.size(0)  # Should be 512
    embedding_dim = old_embeddings.weight.size(1)    # Should be 768
    
    if new_max_length <= old_max_length:
        logger.warning(f"new_max_length {new_max_length} <= existing {old_max_length}, no extension needed")
        return model
    
    logger.info(f"🔧 Extending position embeddings from {old_max_length} to {new_max_length}...")
    
    # Create new position embeddings
    new_embeddings = nn.Embedding(new_max_length, embedding_dim)
    
    # Copy old embeddings (positions 0-511)
    new_embeddings.weight.data[:old_max_length] = old_embeddings.weight.data
    
    # Extend positions 512-1023 using linear interpolation
    # This creates a smooth continuation of the learned position patterns
    # Strategy: Interpolate by cycling through the original embeddings
    for i in range(old_max_length, new_max_length):
        # Map new position to old position space using modulo
        # This effectively "wraps around" the learned patterns
        source_idx = (i - old_max_length) % old_max_length
        
        # Use interpolation for smoother transition
        # Blend between current and next position
        next_idx = (source_idx + 1) % old_max_length
        blend_factor = ((i - old_max_length) % old_max_length) / old_max_length
        
        new_embeddings.weight.data[i] = (
            (1 - blend_factor) * old_embeddings.weight.data[source_idx] + 
            blend_factor * old_embeddings.weight.data[next_idx]
        )
    
    # Replace the position embeddings in the model
    model.distilbert.embeddings.position_embeddings = new_embeddings
    model.config.max_position_embeddings = new_max_length
    
    logger.info(f"✅ Position embeddings extended successfully to {new_max_length}")
    logger.info(f"📊 Embedding shape: [{new_max_length}, {embedding_dim}]")
    logger.info(f"⚠️  Note: Extended positions (512-{new_max_length-1}) are initialized via interpolation")
    logger.info(f"💡 Consider additional training to optimize these new position embeddings")
    
    return model


class DistillationDataCollator:
    """Custom data collator for knowledge distillation that handles both hard and soft labels"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Safely extract tensors, handling both dict and object-like features
        try:
            batch = {
                'input_ids': torch.stack([
                    f['input_ids'] if isinstance(f, dict) else f.input_ids 
                    for f in features
                ]),
                'attention_mask': torch.stack([
                    f['attention_mask'] if isinstance(f, dict) else f.attention_mask 
                    for f in features
                ]),
                'labels': torch.stack([
                    f['labels'] if isinstance(f, dict) else f.labels 
                    for f in features
                ]),
                'soft_labels': torch.stack([
                    f['soft_labels'] if isinstance(f, dict) else f.soft_labels 
                    for f in features
                ])
            }
        except (KeyError, AttributeError) as e:
            logger.error(f"ERROR in data collator: {e}")
            logger.error(f"Feature type: {type(features[0])}")
            if isinstance(features[0], dict):
                logger.error(f"Feature keys: {features[0].keys()}")
            else:
                logger.error(f"Feature attributes: {dir(features[0])}")
            raise
        
        return batch


class ConversationDataset(Dataset):
    """Dataset for conversation classification with knowledge distillation"""
    
    def __init__(self, data: List[Dict], tokenizer: DistilBertTokenizer, 
                 max_length: int, class_order: List[str]):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_order = class_order
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_order)}
        
        logger.info(f"📊 Dataset created with {len(data)} samples")
        logger.info(f"🏷️  Class order: {class_order}")
        logger.info(f"📏 Max sequence length: {max_length} tokens")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Hard label (ground truth)
        hard_label = self.class_to_idx.get(item['hard_label'], 0)
        
        # Soft labels (teacher predictions)
        soft_labels = torch.tensor(item['soft_labels'], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(hard_label, dtype=torch.long),
            'soft_labels': soft_labels,
        }


class WeightedDistillationLoss:
    """Custom loss combining hard labels, soft labels, and confusion penalties"""
    
    def __init__(self, config: DistillationConfig, class_order: List[str]):
        self.config = config
        self.class_order = class_order
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_order)}
        
        self.banned_classes = {'D', 'J', 'M'}
        self.ok_classes = {'A', 'B', 'C' , 'E', 'F', 'G', 'H', 'I', 'K', 'L'}
        
        # Create confusion penalty matrix
        self.penalty_matrix = self._create_penalty_matrix()
        logger.info(f"💥 Penalty matrix created with NSFW/SFW penalty: {config.banned_unbanned_penalty}x")
        
    def _create_penalty_matrix(self) -> torch.Tensor:
        """Create penalty matrix for different types of misclassifications"""
        
        num_classes = len(self.class_order)
        penalty_matrix = torch.ones(num_classes, num_classes)
        
        for i, true_class in enumerate(self.class_order):
            for j, pred_class in enumerate(self.class_order):
                if i == j:
                    penalty_matrix[i, j] = 0.0
                elif self._is_cross_category_error(true_class, pred_class):
                    penalty_matrix[i, j] = self.config.banned_unbanned_penalty
                else:
                    penalty_matrix[i, j] = self.config.within_category_penalty
        
        logger.info("🎯 Penalty Matrix Preview:")
        for i, true_cls in enumerate(self.class_order):
            penalties = [f"{penalty_matrix[i, j].item():.1f}" for j in range(num_classes)]
            logger.info(f"   {true_cls}: {penalties}")
        
        return penalty_matrix
    
    def _is_cross_category_error(self, true_class: str, pred_class: str) -> bool:
        """Check if this is a cross-category (NSFW/SFW) error"""
        true_is_nsfw = true_class in self.banned_classes
        pred_is_nsfw = pred_class in self.banned_classes
        return true_is_nsfw != pred_is_nsfw
    
    def compute_loss(self, student_logits: torch.Tensor, hard_labels: torch.Tensor, 
                    soft_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined distillation loss with confusion penalties"""
        
        device = student_logits.device
        penalty_matrix = self.penalty_matrix.to(device)
        
        # 1. Hard label loss with confusion penalties
        hard_loss = F.cross_entropy(student_logits, hard_labels, reduction='none')
        student_probs = F.softmax(student_logits, dim=-1)
        predicted_classes = torch.argmax(student_probs, dim=-1)
        
        confusion_penalties = penalty_matrix[hard_labels, predicted_classes]
        weighted_hard_loss = (hard_loss * confusion_penalties).mean()
        
        # 2. Soft label loss (KL Divergence)
        student_log_probs = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        teacher_probs = F.softmax(soft_labels / self.config.temperature, dim=-1)
        
        soft_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        # 3. Combined loss
        total_loss = (
            self.config.alpha * soft_loss + 
            (1 - self.config.alpha) * weighted_hard_loss
        )
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'hard_loss': weighted_hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'avg_confusion_penalty': confusion_penalties.mean().item()
        }


class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, loss_fn: WeightedDistillationLoss, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.loss_history = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute distillation loss"""
        
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Compute distillation loss
        loss, loss_dict = self.loss_fn.compute_loss(
            outputs.logits,
            inputs['labels'],
            inputs['soft_labels']
        )
        
        # Store loss components for logging
        self.loss_history.append(loss_dict)
        
        return (loss, outputs) if return_outputs else loss


class DistilBERTDistillation:
    """Main class for DistilBERT knowledge distillation training with 1024 token support"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # 🔥 EXTEND POSITION EMBEDDINGS TO SUPPORT 1024 TOKENS
        if config.max_length > 512:
            logger.info(f"🚀 Extending model to support {config.max_length} tokens...")
            self.model = extend_position_embeddings(self.model, config.max_length)
        
        logger.info(f"✅ DistilBERT model loaded: {config.model_name}")
        logger.info(f"📊 Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_distillation_data(self, data_path: str) -> Tuple[List[Dict], List[str]]:
        """Load knowledge distillation data from a JSON file."""
        
        logger.info(f"Loading distillation data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'distillation_ready' in data and isinstance(data['distillation_ready'], list):
            distillation_samples = data['distillation_ready']
        else:
            raise ValueError(f"Expected a JSON file with a 'distillation_ready' key containing a list of samples in {data_path}")

        if not distillation_samples:
            raise ValueError(f"No samples found in 'distillation_ready' list in {data_path}")

        # Extract class_order from the first sample
        class_order = distillation_samples[0].get('class_order')
        if not class_order:
            raise ValueError("`class_order` not found in the first sample of the data.")

        agreement_rate = distillation_samples[0].get('agreement', 'unknown')

        logger.info(f"📊 Loaded {len(distillation_samples)} distillation samples")
        logger.info(f"🏷️  Classes: {class_order}")
        logger.info(f"📈 Teacher agreement rate from first sample: {agreement_rate}")
        
        return distillation_samples, class_order
    
    def create_datasets(self, samples: List[Dict], class_order: List[str], 
                       train_split: float = 0.8, val_split: float = 0.1, 
                       save_splits: bool = True) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset]:
        """Create train, validation, and test datasets with proper holdout"""
        
        # Shuffle data with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        # Calculate split sizes
        train_size = int(len(samples) * train_split)
        val_size = int(len(samples) * val_split)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        # Save splits to disk
        if save_splits:
            splits_dir = "data/splits"
            os.makedirs(splits_dir, exist_ok=True)
            
            with open(os.path.join(splits_dir, "train.json"), 'w') as f:
                json.dump({"samples": train_samples, "class_order": class_order}, f, indent=2)
            
            with open(os.path.join(splits_dir, "val.json"), 'w') as f:
                json.dump({"samples": val_samples, "class_order": class_order}, f, indent=2)
            
            with open(os.path.join(splits_dir, "test.json"), 'w') as f:
                json.dump({"samples": test_samples, "class_order": class_order}, f, indent=2)
            
            logger.info(f"💾 Splits saved to {splits_dir}/")
        
        # Create datasets
        train_dataset = ConversationDataset(
            train_samples, self.tokenizer, self.config.max_length, class_order
        )
        val_dataset = ConversationDataset(
            val_samples, self.tokenizer, self.config.max_length, class_order
        )
        test_dataset = ConversationDataset(
            test_samples, self.tokenizer, self.config.max_length, class_order
        )
        
        logger.info(f"📚 Train dataset: {len(train_dataset)} samples ({train_split:.1%})")
        logger.info(f"📖 Validation dataset: {len(val_dataset)} samples ({val_split:.1%})")
        logger.info(f"🧪 Test dataset: {len(test_dataset)} samples ({1 - train_split - val_split:.1%})")
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset: ConversationDataset, val_dataset: ConversationDataset,
            output_dir: str = "models/distilbert_distilled"):
        """Train the model with knowledge distillation"""
        
        logger.info("🚀 Starting knowledge distillation training...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create loss function
        loss_fn = WeightedDistillationLoss(self.config, train_dataset.class_order)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            report_to="wandb",
            dataloader_num_workers=0,
            logging_dir=os.path.join(output_dir, "logs"),
            remove_unused_columns=False,  # Critical for custom fields
        )
        
        # Data collator
        data_collator = DistillationDataCollator(self.tokenizer)
        
        # Create trainer
        trainer = DistillationTrainer(
            loss_fn=loss_fn,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        # Test dataset sample
        logger.info("🔍 Testing dataset sample...")
        sample = train_dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Sample types: {[(k, type(v), v.shape if hasattr(v, 'shape') else 'no shape') for k, v in sample.items()]}")

        # Train
        logger.info("🎯 Training started...")
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config_dict = {
            'model_config': {
                'model_name': self.config.model_name,
                'num_labels': self.config.num_labels,
                'max_length': self.config.max_length,
            },
            'training_config': {
                'temperature': self.config.temperature,
                'alpha': self.config.alpha,
                'banned_unbanned_penalty': self.config.banned_unbanned_penalty,
                'within_category_penalty': self.config.within_category_penalty,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
            },
            'class_order': train_dataset.class_order
        }
        
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("✅ Training completed!")
        logger.info(f"💾 Model saved to {output_dir}")
        
        return trainer, train_result
    
    def evaluate_model(self, trainer: DistillationTrainer, dataset: ConversationDataset,
                      class_order: List[str], output_dir: str, prefix: str = "evaluation") -> Dict:
        """Comprehensive model evaluation with confusion matrix"""
        
        logger.info(f"📊 Running comprehensive evaluation on {prefix} set...")
        
        # Get predictions
        predictions = trainer.predict(dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = [sample['labels'].item() for sample in dataset]
        
        # Classification report
        class_names = class_order
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                     output_dict=True, zero_division=0)
        
        logger.info(f"\n📋 {prefix.upper()} CLASSIFICATION REPORT")
        logger.info("="*50)
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'DistilBERT Knowledge Distillation - {prefix.title()} Confusion Matrix\n'
                 f'Accuracy: {report["accuracy"]:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        confusion_path = os.path.join(output_dir, f'{prefix}_confusion_matrix.png')
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 {prefix.title()} confusion matrix saved to {confusion_path}")
        plt.close()
        
        # Calculate business-critical metrics
        banned_classes = {'D', 'J', 'M'}
        ok_classes = {'A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'K', 'L'}
        
        # Cross-category errors
        cross_category_errors = 0
        nsfw_recall_errors = 0
        nsfw_precision_errors = 0
        total_predictions = len(y_true)
        
        for true_idx, pred_idx in zip(y_true, y_pred):
            true_class = class_names[true_idx]
            pred_class = class_names[pred_idx]
            
            true_is_nsfw = true_class in banned_classes
            pred_is_nsfw = pred_class in banned_classes
            
            if true_is_nsfw != pred_is_nsfw:
                cross_category_errors += 1
                if true_is_nsfw and not pred_is_nsfw:
                    nsfw_recall_errors += 1
                elif not true_is_nsfw and pred_is_nsfw:
                    nsfw_precision_errors += 1
        
        cross_category_error_rate = cross_category_errors / total_predictions
        
        logger.info(f"\n🚨 {prefix.upper()} BUSINESS IMPACT ANALYSIS:")
        logger.info(f"   Total predictions: {total_predictions}")
        logger.info(f"   Cross-category errors (NSFW/SFW): {cross_category_errors}")
        logger.info(f"   Cross-category error rate: {cross_category_error_rate:.2%}")
        logger.info(f"   NSFW recall errors (missed NSFW): {nsfw_recall_errors}")
        logger.info(f"   NSFW precision errors (false NSFW): {nsfw_precision_errors}")
        logger.info(f"   Overall accuracy: {report['accuracy']:.2%}")
        
        # Save evaluation results
        eval_results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_order': class_order,
            'cross_category_errors': cross_category_errors,
            'cross_category_error_rate': cross_category_error_rate,
            'nsfw_recall_errors': nsfw_recall_errors,
            'nsfw_precision_errors': nsfw_precision_errors,
            'total_predictions': total_predictions,
            'evaluation_type': prefix
        }
        
        eval_path = os.path.join(output_dir, f'{prefix}_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"💾 {prefix.title()} results saved to {eval_path}")
        
        return eval_results
    
    def export_to_onnx(self, model_dir: str, class_order: List[str]):
        """Export the model to ONNX format"""
        
        logger.info(f"🚀 Exporting model to ONNX format...")
        
        try:
            # Load the trained model and tokenizer
            model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            
            # IMPORTANT: Re-extend position embeddings for ONNX export
            if self.config.max_length > 512:
                logger.info(f"🔧 Re-extending position embeddings for ONNX export...")
                model = extend_position_embeddings(model, self.config.max_length)
            
            # Create a dummy input for tracing
            dummy_text = "This is a sample text for ONNX export."
            dummy_input = tokenizer(
                dummy_text, 
                return_tensors='pt', 
                max_length=self.config.max_length, 
                padding='max_length', 
                truncation=True
            )
            
            onnx_path = os.path.join(model_dir, "model.onnx")
            
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            logger.info(f"✅ ONNX model exported successfully to {onnx_path}")
            
            # Save class order with ONNX model for inference
            onnx_config_path = os.path.join(model_dir, 'onnx_config.json')
            onnx_config = {
                'class_order': class_order,
                'max_length': self.config.max_length
            }
            with open(onnx_config_path, 'w') as f:
                json.dump(onnx_config, f, indent=2)
            
            logger.info(f"💾 ONNX configuration with class order saved to {onnx_config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export to ONNX: {e}")
            if wandb.run:
                wandb.log({"onnx_export_error": str(e)})


def safe_log_artifact(path, name, artifact_type):
    """Log artifact only if file exists"""
    if os.path.exists(path):
        wandb.log_artifact(path, name=name, type=artifact_type)
        logger.info(f"✅ Logged artifact: {name}")
    else:
        logger.warning(f"⚠️  Artifact not found, skipping: {path}")


async def main():
    """Main training function with immediate post-training evaluation"""
    
    # Configuration
    DATA_PATH = os.getenv("DISTILLATION_DATA_PATH", "data/agreed_distillation_data.json")
    OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "models/distilbert_distilled_1024")
    
    # Initialize wandb
    wandb.init(
        project="distilbert-conversation-classifier-1024-tokens",
        name=f"distilbert-1024-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "data_path": DATA_PATH,
            "output_dir": OUTPUT_DIR,
            "max_length": 1024,
        }
    )
    
    config = DistillationConfig(
        batch_size=int(os.getenv("BATCH_SIZE", "8")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
        num_epochs=int(os.getenv("NUM_EPOCHS", "10")),
        temperature=float(os.getenv("TEMPERATURE", "4.0")),
        alpha=float(os.getenv("ALPHA", "0.7")),
        banned_unbanned_penalty=float(os.getenv("NSFW_SFW_PENALTY", "5.0")),
    )
    
    logger.info("🎓 DistilBERT Knowledge Distillation Training (1024 Tokens)")
    logger.info(f"📊 Temperature: {config.temperature}")
    logger.info(f"⚖️  Alpha (soft/hard loss balance): {config.alpha}")
    logger.info(f"💥 NSFW/SFW penalty: {config.banned_unbanned_penalty}x")
    logger.info(f"📏 Max sequence length: {config.max_length} tokens")
    logger.info(f"📂 Data: {DATA_PATH}")
    logger.info(f"📁 Output: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize distillation trainer
    distiller = DistilBERTDistillation(config)
    
    # Load data
    samples, class_order = distiller.load_distillation_data(DATA_PATH)
    
    # Create datasets with proper train/val/test split
    train_dataset, val_dataset, test_dataset = distiller.create_datasets(samples, class_order)
    
    # Train model
    logger.info("🚀 Starting training phase...")
    trainer, train_result = distiller.train(train_dataset, val_dataset, OUTPUT_DIR)
    
    # Immediate post-training evaluation
    logger.info("📊 Running immediate post-training evaluation...")
    
    # 1. Validation set evaluation
    logger.info("📖 Evaluating on validation set...")
    val_results = distiller.evaluate_model(trainer, val_dataset, class_order, OUTPUT_DIR, prefix="validation")
    
    # 2. Test set evaluation
    logger.info("🧪 Evaluating on held-out test set...")
    test_results = distiller.evaluate_model(trainer, test_dataset, class_order, OUTPUT_DIR, prefix="test")
    
    # 3. Compare validation vs test performance
    val_accuracy = val_results['classification_report']['accuracy']
    test_accuracy = test_results['classification_report']['accuracy']
    accuracy_drop = val_accuracy - test_accuracy
    
    val_cross_error = val_results['cross_category_error_rate']
    test_cross_error = test_results['cross_category_error_rate']
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETION SUMMARY")
    logger.info("="*60)
    logger.info(f"🎯 Validation Accuracy: {val_accuracy:.3f}")
    logger.info(f"🧪 Test Accuracy: {test_accuracy:.3f}")
    logger.info(f"📉 Accuracy Drop: {accuracy_drop:.3f} ({'⚠️  HIGH' if accuracy_drop > 0.05 else '✅ OK'})")
    logger.info(f"🚨 Validation Cross-Category Errors: {val_cross_error:.3f}")
    logger.info(f"🚨 Test Cross-Category Errors: {test_cross_error:.3f}")
    
    # Overfitting warning
    if accuracy_drop > 0.05:
        logger.warning("⚠️  Significant accuracy drop detected - possible overfitting!")
        logger.warning("💡 Consider: reducing epochs, increasing regularization, or more data")
    
    # Business readiness assessment
    test_acceptable = test_cross_error < 0.05 and test_accuracy > 0.75
    logger.info(f"🏭 Production Readiness: {'✅ READY' if test_acceptable else '⚠️  NEEDS IMPROVEMENT'}")
    
    # Export to ONNX
    distiller.export_to_onnx(OUTPUT_DIR, class_order)
    
    # Save comprehensive training summary
    training_summary = {
        'training_config': {
            'temperature': config.temperature,
            'alpha': config.alpha,
            'banned_unbanned_penalty': config.banned_unbanned_penalty,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'max_length': config.max_length,
        },
        'data_splits': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        },
        'performance_summary': {
            'validation': {
                'accuracy': val_accuracy,
                'cross_category_error_rate': val_cross_error
            },
            'test': {
                'accuracy': test_accuracy,
                'cross_category_error_rate': test_cross_error
            },
            'overfitting_metrics': {
                'accuracy_drop': accuracy_drop,
                'overfitting_detected': accuracy_drop > 0.05
            }
        },
        'production_assessment': {
            'ready_for_production': test_acceptable,
            'accuracy_threshold_met': test_accuracy > 0.75,
            'cross_category_threshold_met': test_cross_error < 0.05
        },
        'class_order': class_order,
        'training_completion_time': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'training_completion_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info("✅ Training pipeline complete!")
    logger.info(f"📊 Comprehensive summary saved to: {summary_path}")
    logger.info("🎯 Model ready for production deployment!")
    
    # Log artifacts to wandb
    wandb.log({"training_summary": training_summary})
    safe_log_artifact(summary_path, "training_summary", "results")
    safe_log_artifact(os.path.join(OUTPUT_DIR, "validation_confusion_matrix.png"), "validation_confusion_matrix", "image")
    safe_log_artifact(os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"), "test_confusion_matrix", "image")
    safe_log_artifact(os.path.join(OUTPUT_DIR, "validation_results.json"), "validation_results", "results")
    safe_log_artifact(os.path.join(OUTPUT_DIR, "test_results.json"), "test_results", "results")
    safe_log_artifact(os.path.join(OUTPUT_DIR, "onnx_config.json"), "onnx_config", "config")
    
    # Log model artifact
    model_artifact = wandb.Artifact(
        "distilbert-distilled-model-1024", 
        type="model",
        description="Distilled DistilBERT model (1024 tokens) for conversation classification"
    )
    model_artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(model_artifact)
    
    wandb.finish()
    
    return training_summary


if __name__ == "__main__":
    try:
        import torch
        import transformers
        import sklearn
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install torch transformers scikit-learn matplotlib seaborn")
        exit(1)
    
    import asyncio
    asyncio.run(main())