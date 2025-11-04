#!/usr/bin/env python3
"""
Comprehensive Model Comparison Evaluation

Compares PyTorch, Quantized ONNX, and Llama 3.1 8B (Groq) models on:
- Overall accuracy
- Per-class performance
- Cross-category errors (banned/unbanned confusion)
- Inference speed
- Cost (for API models)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    """Results for a single model"""
    name: str
    predictions: List[str]
    confidences: List[float]
    inference_times: List[float]
    total_time: float
    accuracy: float
    cross_category_errors: int
    cross_category_error_rate: float
    banned_recall_errors: int  # Banned labeled as unbanned (HIGH RISK)
    banned_precision_errors: int  # Unbanned labeled as banned (revenue loss)
    per_class_metrics: Dict
    confusion_matrix: np.ndarray
    cost: float = 0.0  # For API models

class ModelEvaluator:
    """Evaluates and compares different model implementations"""
    
    def __init__(self, test_data_path: str, class_order: List[str]):
        self.test_data_path = Path(test_data_path)
        self.class_order = class_order
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_order)}
        
        # Define banned/unbanned categories
        self.banned_classes = {'D', 'J', 'M'}
        self.unbanned_classes = {'A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'K', 'L'}
        
        logger.info(f"🏷️  Classes: {class_order}")
        logger.info(f"🚨 Banned classes: {self.banned_classes}")
        logger.info(f"✅ Unbanned classes: {self.unbanned_classes}")
    
    def load_test_data(self) -> Tuple[List[str], List[str]]:
        """Load test data and extract texts and labels"""
        
        logger.info(f"📂 Loading test data from {self.test_data_path}")
        
        with open(self.test_data_path) as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'samples' in data:
            samples = data['samples']
        else:
            samples = data
        
        texts = [s['text'] for s in samples]
        true_labels = [s['hard_label'] for s in samples]
        
        logger.info(f"📊 Loaded {len(texts)} test samples")
        
        return texts, true_labels
    
    def evaluate_pytorch_model(self, model_path: str, texts: List[str], 
                               true_labels: List[str]) -> ModelResult:
        """Evaluate PyTorch model (safetensors)"""
        
        logger.info("\n" + "="*70)
        logger.info("🔥 EVALUATING PYTORCH MODEL")
        logger.info("="*70)
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            raise
        
        # Load model
        logger.info(f"📂 Loading PyTorch model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"✅ Model loaded on {device}")
        
        # Run inference
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        
        start_total = time.perf_counter()
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(texts)}")
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            confidence = probs.max().item()
            
            predictions.append(self.class_order[pred_idx])
            confidences.append(confidence)
            inference_times.append(inference_time)
        
        total_time = time.perf_counter() - start_total
        
        logger.info(f"✅ PyTorch inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        
        return self._compute_metrics("PyTorch (safetensors)", predictions, confidences, 
                                    inference_times, total_time, true_labels)
    
    def evaluate_onnx_model(self, model_path: str, texts: List[str], 
                           true_labels: List[str]) -> ModelResult:
        """Evaluate quantized ONNX model"""
        
        logger.info("\n" + "="*70)
        logger.info("⚡ EVALUATING QUANTIZED ONNX MODEL")
        logger.info("="*70)
        
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            raise
        
        model_dir = Path(model_path).parent
        
        # Load tokenizer
        logger.info(f"📂 Loading ONNX model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        logger.info(f"✅ ONNX model loaded")
        
        # Run inference
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        
        start_total = time.perf_counter()
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(texts)}")
            
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='np'
            )
            
            start_time = time.perf_counter()
            outputs = session.run(
                None,
                {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
            )
            inference_time = (time.perf_counter() - start_time) * 1000
            
            logits = outputs[0]
            probs = self._softmax(logits[0])
            pred_idx = np.argmax(probs)
            confidence = np.max(probs)
            
            predictions.append(self.class_order[pred_idx])
            confidences.append(float(confidence))
            inference_times.append(inference_time)
        
        total_time = time.perf_counter() - start_total
        
        logger.info(f"✅ ONNX inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        
        return self._compute_metrics("Quantized ONNX", predictions, confidences, 
                                    inference_times, total_time, true_labels)
    
    async def evaluate_groq_llama(self, texts: List[str], true_labels: List[str], 
                                 api_key: str) -> ModelResult:
        """Evaluate Llama 3.1 8B Instant via Groq API"""
        
        logger.info("\n" + "="*70)
        logger.info("🦙 EVALUATING LLAMA 3.1 8B INSTANT (GROQ)")
        logger.info("="*70)
        
        try:
            from groq import AsyncGroq
        except ImportError:
            logger.error("❌ Missing groq package. Install with: pip install groq")
            raise
        
        client = AsyncGroq(api_key=api_key)
        
        # Create classification prompt
        system_prompt = f"""You are a conversation classifier. Classify conversations into exactly one of these categories:

{', '.join(self.class_order)}

Respond with ONLY the single letter category (A-M), nothing else."""
        
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        logger.info("⚠️  This will take several minutes due to API rate limits...")
        
        start_total = time.perf_counter()
        
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(texts)}")
            
            # Truncate text to avoid token limits
            truncated_text = text[:2000] if len(text) > 2000 else text
            
            start_time = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Classify this conversation:\n\n{truncated_text}"}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                inference_time = (time.perf_counter() - start_time) * 1000
                
                prediction = response.choices[0].message.content.strip().upper()
                
                # Validate prediction
                if prediction not in self.class_order:
                    logger.warning(f"Invalid prediction '{prediction}' at index {i}, defaulting to A")
                    prediction = 'A'
                
                predictions.append(prediction)
                confidences.append(1.0)  # Groq doesn't provide confidence scores
                inference_times.append(inference_time)
                
                # Rate limiting - be conservative
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error at index {i}: {e}")
                predictions.append('A')  # Default fallback
                confidences.append(0.0)
                inference_times.append(0.0)
        
        total_time = time.perf_counter() - start_total
        
        # Calculate cost (Llama 3.1 8B Instant pricing)
        # Approximate: $0.05 per 1M input tokens, $0.08 per 1M output tokens
        avg_input_tokens = 500  # Rough estimate
        avg_output_tokens = 5
        total_input_tokens = len(texts) * avg_input_tokens
        total_output_tokens = len(texts) * avg_output_tokens
        
        cost = (total_input_tokens / 1_000_000 * 0.05 + 
                total_output_tokens / 1_000_000 * 0.08)
        
        logger.info(f"✅ Groq inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        logger.info(f"💰 Estimated cost: ${cost:.4f}")
        
        result = self._compute_metrics("Llama 3.1 8B (Groq)", predictions, confidences, 
                                      inference_times, total_time, true_labels)
        result.cost = cost
        return result
    
    def _compute_metrics(self, model_name: str, predictions: List[str], 
                        confidences: List[float], inference_times: List[float],
                        total_time: float, true_labels: List[str]) -> ModelResult:
        """Compute comprehensive metrics for a model"""
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Cross-category errors
        cross_errors = 0
        banned_recall_errors = 0
        banned_precision_errors = 0
        
        for true_label, pred_label in zip(true_labels, predictions):
            true_is_banned = true_label in self.banned_classes
            pred_is_banned = pred_label in self.banned_classes
            
            if true_is_banned != pred_is_banned:
                cross_errors += 1
                if true_is_banned and not pred_is_banned:
                    banned_recall_errors += 1
                elif not true_is_banned and pred_is_banned:
                    banned_precision_errors += 1
        
        cross_error_rate = cross_errors / len(true_labels)
        
        # Per-class metrics
        report = classification_report(
            true_labels, predictions, 
            target_names=self.class_order,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            true_labels, predictions,
            labels=self.class_order
        )
        
        return ModelResult(
            name=model_name,
            predictions=predictions,
            confidences=confidences,
            inference_times=inference_times,
            total_time=total_time,
            accuracy=accuracy,
            cross_category_errors=cross_errors,
            cross_category_error_rate=cross_error_rate,
            banned_recall_errors=banned_recall_errors,
            banned_precision_errors=banned_precision_errors,
            per_class_metrics=report,
            confusion_matrix=cm
        )
    
    def _softmax(self, x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def generate_comparison_report(self, results: List[ModelResult], 
                                  output_path: str):
        """Generate comprehensive comparison report"""
        
        logger.info("\n" + "="*70)
        logger.info("📊 COMPARISON REPORT")
        logger.info("="*70)
        
        # Print comparison table
        print("\n" + "="*100)
        print(f"{'Model':<30} {'Accuracy':<12} {'Cross-Cat Err':<15} {'Banned→Safe':<15} {'Avg Time (ms)':<15} {'Cost':<10}")
        print("="*100)
        
        for result in results:
            print(f"{result.name:<30} "
                  f"{result.accuracy*100:>10.2f}%  "
                  f"{result.cross_category_error_rate*100:>13.2f}%  "
                  f"{result.banned_recall_errors:>13d}  "
                  f"{np.mean(result.inference_times):>13.2f}  "
                  f"${result.cost:>8.4f}")
        
        print("="*100)
        
        # Detailed metrics for each model
        for result in results:
            print(f"\n{'='*70}")
            print(f"DETAILED METRICS: {result.name}")
            print(f"{'='*70}")
            print(f"Overall Accuracy: {result.accuracy:.3f}")
            print(f"Total Samples: {len(result.predictions)}")
            print(f"Total Time: {result.total_time:.2f}s")
            print(f"\nSpeed Metrics:")
            print(f"  Average: {np.mean(result.inference_times):.2f}ms")
            print(f"  Min: {np.min(result.inference_times):.2f}ms")
            print(f"  Max: {np.max(result.inference_times):.2f}ms")
            print(f"  P95: {np.percentile(result.inference_times, 95):.2f}ms")
            print(f"\nCross-Category Errors:")
            print(f"  Total: {result.cross_category_errors} ({result.cross_category_error_rate:.2%})")
            print(f"  Banned→Unbanned (HIGH RISK): {result.banned_recall_errors}")
            print(f"  Unbanned→Banned (revenue loss): {result.banned_precision_errors}")
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'models': []
        }
        
        for result in results:
            report_data['models'].append({
                'name': result.name,
                'accuracy': float(result.accuracy),
                'cross_category_error_rate': float(result.cross_category_error_rate),
                'cross_category_errors': result.cross_category_errors,
                'banned_recall_errors': result.banned_recall_errors,
                'banned_precision_errors': result.banned_precision_errors,
                'avg_inference_time_ms': float(np.mean(result.inference_times)),
                'p95_inference_time_ms': float(np.percentile(result.inference_times, 95)),
                'total_time_s': float(result.total_time),
                'cost_usd': float(result.cost),
                'per_class_metrics': result.per_class_metrics,
                'confusion_matrix': result.confusion_matrix.tolist()
            })
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n💾 Report saved to {output_path}")

async def main():
    """Main evaluation function"""
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    TEST_DATA_PATH = "data/splits/test.json"
    PYTORCH_MODEL_PATH = "models/distilbert_distilled_alpha_0.0"
    ONNX_MODEL_PATH = "models/distilbert_distilled_alpha_0.0/quantized/model.onnx"
    OUTPUT_PATH = "eval/comparison_results.json"
    
    # Get class order from training config
    with open(Path(PYTORCH_MODEL_PATH) / "training_completion_summary.json") as f:
        config = json.load(f)
        class_order = config['class_order']
    
    logger.info("🚀 MODEL COMPARISON EVALUATION")
    logger.info("="*70)
    logger.info(f"📂 Test data: {TEST_DATA_PATH}")
    logger.info(f"🔥 PyTorch model: {PYTORCH_MODEL_PATH}")
    logger.info(f"⚡ ONNX model: {ONNX_MODEL_PATH}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(TEST_DATA_PATH, class_order)
    
    # Load test data
    texts, true_labels = evaluator.load_test_data()
    
    # Evaluate models
    results = []
    
    # 1. PyTorch model
    try:
        pytorch_result = evaluator.evaluate_pytorch_model(
            PYTORCH_MODEL_PATH, texts, true_labels
        )
        results.append(pytorch_result)
    except Exception as e:
        logger.error(f"❌ PyTorch evaluation failed: {e}")
    
    # 2. ONNX model
    try:
        onnx_result = evaluator.evaluate_onnx_model(
            ONNX_MODEL_PATH, texts, true_labels
        )
        results.append(onnx_result)
    except Exception as e:
        logger.error(f"❌ ONNX evaluation failed: {e}")
    
    # 3. Groq Llama (optional - can be slow and costly)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            logger.info("\n⚠️  Groq API evaluation will take several minutes...")
            user_input = input("Proceed with Groq evaluation? (y/n): ")
            if user_input.lower() == 'y':
                groq_result = await evaluator.evaluate_groq_llama(
                    texts, true_labels, groq_api_key
                )
                results.append(groq_result)
        except Exception as e:
            logger.error(f"❌ Groq evaluation failed: {e}")
    else:
        logger.warning("⚠️  GROQ_API_KEY not found, skipping Groq evaluation")
    
    # Generate comparison report
    if results:
        evaluator.generate_comparison_report(results, OUTPUT_PATH)
        logger.info("\n✅ Evaluation complete!")
    else:
        logger.error("❌ No successful evaluations")

if __name__ == "__main__":
    asyncio.run(main())