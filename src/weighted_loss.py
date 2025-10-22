#!/usr/bin/env python3
"""
Weighted loss implementation for intent classification with asymmetric penalties
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import Trainer
import logging

logger = logging.getLogger(__name__)

class LossWeightConfig:
    """Configuration for asymmetric loss weights in intent classification"""
    
    def __init__(self, intent_labels: List[str]):
        self.intent_labels = intent_labels
        self.label_to_id = {label: idx for idx, label in enumerate(intent_labels)}
        self.num_classes = len(intent_labels)
        
        # Define high-risk categories that need strong penalties when misclassified
        self.high_risk_categories = {
            "relationships_and_personal_reflection",
            "games_and_role_play", 
            "creative_ideation"
        }
        
        # Define low-risk categories where mistakes are less costly
        self.low_risk_categories = {
            "greetings_and_chitchat",
            "how_to_advice",
            "specific_info"
        }
        
        # Create the weight matrix
        self.weight_matrix = self._create_weight_matrix()
        
        # Log examples after matrix is created
        self._log_weight_examples()
    
    def _create_weight_matrix(self) -> torch.Tensor:
        """
        Create pred_category x true_category weight matrix
        
        High weights = high penalty for that mistake
        Matrix is [predicted_class, true_class]
        """
        # Start with default weight of 1.0 for all mistakes
        weights = torch.ones(self.num_classes, self.num_classes)
        
        for true_idx, true_label in enumerate(self.intent_labels):
            for pred_idx, pred_label in enumerate(self.intent_labels):
                # Correct predictions get weight 1.0 (standard)
                if true_idx == pred_idx:
                    weights[pred_idx, true_idx] = 1.0
                else:
                    # Determine penalty based on mistake type
                    weights[pred_idx, true_idx] = self._get_mistake_penalty(
                        true_label, pred_label
                    )
        
        logger.info(f"Created weight matrix with shape {weights.shape}")
        return weights
    
    def _get_mistake_penalty(self, true_label: str, pred_label: str) -> float:
        """
        Determine penalty weight for a specific mistake
        
        Args:
            true_label: Actual intent category  
            pred_label: Predicted intent category
            
        Returns:
            Weight multiplier for loss (higher = more penalty)
        """
        # CRITICAL MISTAKE: High-risk content classified as low-risk
        # This is the worst case - NSFW content getting ads
        if (true_label in self.high_risk_categories and 
            pred_label in self.low_risk_categories):
            return 10.0  # Very high penalty
        
        # HIGH MISTAKE: High-risk content misclassified as medium-risk
        if (true_label in self.high_risk_categories and 
            pred_label not in self.high_risk_categories):
            return 5.0   # High penalty
        
        # MEDIUM MISTAKE: Any content classified as high-risk when it's not
        # (False positive for NSFW - less bad than false negative)
        if (true_label not in self.high_risk_categories and 
            pred_label in self.high_risk_categories):
            return 2.0   # Medium penalty
        
        # LOW MISTAKE: Confusion between low-risk categories
        if (true_label in self.low_risk_categories and 
            pred_label in self.low_risk_categories):
            return 0.5   # Low penalty - we don't care much
        
        # DEFAULT: Standard penalty for other mistakes
        return 1.0
    
    def _log_weight_examples(self):
        """Log some example weights for verification"""
        examples = [
            ("relationships_and_personal_reflection", "greetings_and_chitchat"),
            ("games_and_role_play", "how_to_advice"), 
            ("creative_ideation", "specific_info"),
            ("greetings_and_chitchat", "relationships_and_personal_reflection"),
            ("how_to_advice", "greetings_and_chitchat"),
        ]
        
        logger.info("Example loss weights:")
        for true_label, pred_label in examples:
            if true_label in self.label_to_id and pred_label in self.label_to_id:
                true_idx = self.label_to_id[true_label]
                pred_idx = self.label_to_id[pred_label]
                weight = self.weight_matrix[pred_idx, true_idx].item()
                logger.info(f"  True: {true_label[:20]:<20} | Pred: {pred_label[:20]:<20} | Weight: {weight}")

class WeightedLossTrainer(Trainer):
    """Custom Trainer with asymmetric loss weighting"""
    
    def __init__(self, loss_config: LossWeightConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config
        self.weight_matrix = loss_config.weight_matrix.to(self.args.device)
        
        # Track loss statistics
        self.total_weighted_loss = 0.0
        self.total_standard_loss = 0.0
        self.loss_call_count = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function with asymmetric weighting
        
        Args:
            model: The model being trained
            inputs: Batch of input data
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)
            
        Returns:
            Loss tensor (and optionally model outputs)
        """
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if labels is not None:
            # Compute standard cross-entropy loss (unreduced)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            standard_loss = loss_fct(logits, labels)  # Shape: [batch_size]
            
            # Get predicted classes
            predicted_classes = torch.argmax(logits, dim=-1)  # Shape: [batch_size]
            
            # Look up weights for each sample in batch
            # weight_matrix[pred_class, true_class]
            batch_weights = self.weight_matrix[predicted_classes, labels]  # Shape: [batch_size]
            
            # Apply weights and compute final loss
            weighted_loss = standard_loss * batch_weights
            final_loss = weighted_loss.mean()
            
            # Track statistics
            self._update_loss_stats(standard_loss.mean(), final_loss)
            
            # Optional: Log occasional examples for debugging
            if self.loss_call_count % 100 == 0:
                self._log_batch_examples(labels, predicted_classes, batch_weights, standard_loss, weighted_loss)
            
        else:
            final_loss = outputs.loss
        
        return (final_loss, outputs) if return_outputs else final_loss
    
    def _update_loss_stats(self, standard_loss: torch.Tensor, weighted_loss: torch.Tensor):
        """Track loss statistics for monitoring"""
        self.total_standard_loss += standard_loss.item()
        self.total_weighted_loss += weighted_loss.item()
        self.loss_call_count += 1
        
        # Log every 500 steps
        if self.loss_call_count % 500 == 0:
            avg_standard = self.total_standard_loss / self.loss_call_count
            avg_weighted = self.total_weighted_loss / self.loss_call_count
            multiplier = avg_weighted / avg_standard if avg_standard > 0 else 1.0
            
            logger.info(f"Loss Stats (Step {self.loss_call_count}):")
            logger.info(f"  Avg Standard Loss: {avg_standard:.4f}")
            logger.info(f"  Avg Weighted Loss: {avg_weighted:.4f}")
            logger.info(f"  Loss Multiplier: {multiplier:.2f}x")
    
    def _log_batch_examples(self, true_labels, pred_labels, weights, standard_losses, weighted_losses):
        """Log a few examples from current batch for debugging"""
        logger.info("Batch Loss Examples:")
        
        # Show first 3 examples
        for i in range(min(3, len(true_labels))):
            true_idx = true_labels[i].item()
            pred_idx = pred_labels[i].item()
            weight = weights[i].item()
            std_loss = standard_losses[i].item()
            weighted_loss = weighted_losses[i].item()
            
            true_label = self.loss_config.intent_labels[true_idx] if true_idx < len(self.loss_config.intent_labels) else f"idx_{true_idx}"
            pred_label = self.loss_config.intent_labels[pred_idx] if pred_idx < len(self.loss_config.intent_labels) else f"idx_{pred_idx}"
            
            logger.info(f"  [{i}] True: {true_label[:15]:<15} | Pred: {pred_label[:15]:<15} | "
                       f"Weight: {weight:.1f} | Loss: {std_loss:.3f} -> {weighted_loss:.3f}")

def create_weighted_trainer(loss_config: LossWeightConfig, model, args, train_dataset, 
                          eval_dataset=None, tokenizer=None, compute_metrics=None, callbacks=None):
    """
    Factory function to create a WeightedLossTrainer
    
    Args:
        loss_config: LossWeightConfig instance
        model: The model to train
        args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        tokenizer: Tokenizer
        compute_metrics: Optional metrics function
        callbacks: Optional callbacks list
        
    Returns:
        WeightedLossTrainer instance
    """
    trainer_kwargs = {
        "loss_config": loss_config,
        "model": model,
        "args": args, 
        "train_dataset": train_dataset,
        "tokenizer": tokenizer,
    }
    
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks
        
    return WeightedLossTrainer(**trainer_kwargs)