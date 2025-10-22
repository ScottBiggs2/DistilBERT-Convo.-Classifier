#!/usr/bin/env python3
"""
Export trained DistilBERT model to ONNX format with quantization for fast CPU inference
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoOptimizationConfig
import onnxruntime as ort
from typing import Dict, List
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(
    model_path: str, 
    output_path: str, 
    quantize: bool = True,
    optimize: bool = True
):
    """
    Export PyTorch model to ONNX format with optional quantization and optimization
    """
    logger.info(f"Loading model from {model_path}")
    
    # Verify input model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Create output directory
    onnx_path = Path(output_path)
    onnx_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Converting to ONNX format...")
    
    # Convert to ONNX - this handles the conversion automatically
    try:
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            model_path, 
            export=True
        )
        logger.info("Successfully converted to ONNX")
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        raise
    
    # Save tokenizer to output path
    tokenizer.save_pretrained(onnx_path)
    
    if optimize:
        logger.info("Optimizing ONNX model...")
        try:
            optimizer = ORTOptimizer.from_pretrained(onnx_model)
            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level="all"
            )
            optimizer.optimize(save_dir=onnx_path, optimization_config=optimization_config)
            onnx_model = ORTModelForSequenceClassification.from_pretrained(onnx_path)
            logger.info("Model optimization completed")
        except Exception as e:
            logger.warning(f"Optimization failed: {e}. Continuing without optimization.")
            optimize = False
    
    if quantize:
        logger.info("Quantizing ONNX model for CPU inference...")
        try:
            quantizer = ORTQuantizer.from_pretrained(onnx_model)
            
            # Try different quantization configs in order of preference
            quantization_configs = [
                ("AVX512_VNNI", AutoQuantizationConfig.avx512_vnni(is_static=False)),
                ("AVX2", AutoQuantizationConfig.avx2(is_static=False)),
                ("ARM64", AutoQuantizationConfig.arm64(is_static=False))
            ]
            
            quantized = False
            for config_name, config in quantization_configs:
                try:
                    quantizer.quantize(save_dir=onnx_path, quantization_config=config)
                    logger.info(f"Model quantized successfully using {config_name}")
                    quantized = True
                    break
                except Exception as e:
                    logger.warning(f"Quantization with {config_name} failed: {e}")
                    continue
            
            if not quantized:
                logger.warning("All quantization methods failed. Saving unquantized model.")
                onnx_model.save_pretrained(onnx_path)
                quantize = False
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Saving unquantized model.")
            onnx_model.save_pretrained(onnx_path)
            quantize = False
    else:
        # Save without quantization
        onnx_model.save_pretrained(onnx_path)
    
    # Save model configuration
    config_info = {
        "model_type": "distilbert-intent-classifier",
        "quantized": quantize,
        "optimized": optimize,
        "max_length": 512,
        "num_labels": len(model.config.id2label) if hasattr(model.config, 'id2label') else 18,
        "label_mapping": dict(model.config.id2label) if hasattr(model.config, 'id2label') else {},
        "export_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(onnx_path / "model_config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    
    logger.info(f"ONNX model saved to {onnx_path}")
    return onnx_path, config_info

def benchmark_model(onnx_path: str, num_samples: int = 100):
    """
    Benchmark the ONNX model to measure inference speed
    """
    logger.info(f"Benchmarking model with {num_samples} samples...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    ort_model = ORTModelForSequenceClassification.from_pretrained(onnx_path)
    
    # Sample conversation texts for benchmarking
    sample_texts = [
        "[USER] How do I solve this math problem step by step? [ASSISTANT] I'll help you solve it step by step.",
        "[USER] Can you help me write an email to my boss? [ASSISTANT] I'll help you draft a professional email.",
        "[USER] Write a short story about a dragon [ASSISTANT] Here's a creative story about a dragon...",
        "[USER] What's the best laptop for programming? [ASSISTANT] I'd recommend these programming laptops...",
        "[USER] Translate this to French please [ASSISTANT] I'll translate that to French for you.",
        "[USER] I'm feeling anxious about my relationship [ASSISTANT] I understand you're feeling anxious...",
        "[USER] Generate marketing ideas for my startup [ASSISTANT] Here are some creative marketing ideas...",
        "[USER] What ingredients do I need for lasagna? [ASSISTANT] Here are the ingredients for lasagna...",
        "[USER] Explain photosynthesis in simple terms [ASSISTANT] Photosynthesis is the process where plants...",
        "[USER] Hi there, how are you doing today? [ASSISTANT] Hello! I'm doing well, thank you for asking."
    ] * (num_samples // 10 + 1)
    
    sample_texts = sample_texts[:num_samples]
    
    # Tokenize all samples
    encodings = tokenizer(
        sample_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Warm up
    logger.info("Warming up model...")
    for _ in range(5):
        _ = ort_model(**{k: v[:1] for k, v in encodings.items()})
    
    # Benchmark single predictions (most realistic for API)
    times = []
    logger.info("Running single prediction benchmark...")
    
    for i in range(min(50, num_samples)):
        single_encoding = {k: v[i:i+1] for k, v in encodings.items()}
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = ort_model(**single_encoding)
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Benchmark batch predictions
    batch_times = []
    batch_sizes = [1, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        if batch_size > num_samples:
            continue
            
        batch_encoding = {k: v[:batch_size] for k, v in encodings.items()}
        
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = ort_model(**batch_encoding)
        end_time = time.perf_counter()
        
        batch_time = (end_time - start_time) * 1000
        per_sample_time = batch_time / batch_size
        batch_times.append((batch_size, batch_time, per_sample_time))
    
    # Results
    avg_single_time = np.mean(times)
    p95_single_time = np.percentile(times, 95)
    p99_single_time = np.percentile(times, 99)
    
    print("\n" + "="*50)
    print("ONNX MODEL BENCHMARK RESULTS")
    print("="*50)
    print(f"Single Prediction Performance:")
    print(f"  Average: {avg_single_time:.2f}ms")
    print(f"  P95: {p95_single_time:.2f}ms") 
    print(f"  P99: {p99_single_time:.2f}ms")
    print(f"  Min: {min(times):.2f}ms")
    print(f"  Max: {max(times):.2f}ms")
    
    print(f"\nBatch Performance:")
    for batch_size, total_time, per_sample in batch_times:
        print(f"  Batch size {batch_size:2d}: {total_time:6.2f}ms total, {per_sample:5.2f}ms per sample")
    
    # Check if we meet target
    target_met = avg_single_time <= 50
    print(f"\nTarget Performance (≤50ms): {'✅ MET' if target_met else '❌ NOT MET'}")
    
    return {
        "avg_single_time_ms": avg_single_time,
        "p95_single_time_ms": p95_single_time,
        "p99_single_time_ms": p99_single_time,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "batch_performance": batch_times,
        "target_met": target_met,
        "benchmark_timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

def test_inference(onnx_path: str):
    """
    Test the exported ONNX model with sample predictions
    """
    logger.info("Testing ONNX model inference...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    ort_model = ORTModelForSequenceClassification.from_pretrained(onnx_path)
    
    # Load label mapping
    config_path = Path(onnx_path) / "model_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        id_to_label = {int(k): v for k, v in config["label_mapping"].items()}
    else:
        # Fallback to default labels
        id_to_label = {i: f"label_{i}" for i in range(18)}
    
    # Test samples
    test_samples = [
        "[USER] How do I solve this calculus problem step by step? [ASSISTANT] I'll help you solve it.",
        "[USER] Can you help me write a professional email to my manager? [ASSISTANT] I'll help you draft it.",
        "[USER] Write a creative short story about time travel [ASSISTANT] Here's a story about time travel...",
        "[USER] What's the best gaming laptop under $1500? [ASSISTANT] I'd recommend these laptops...",
        "[USER] Translate 'Hello, how are you?' to Spanish [ASSISTANT] That translates to 'Hola, ¿cómo estás?'",
        "[USER] I'm struggling with anxiety in my relationship [ASSISTANT] I understand this is difficult...",
        "[USER] Generate creative names for my new coffee shop [ASSISTANT] Here are some creative names...",
        "[USER] What ingredients do I need to make authentic Italian pasta? [ASSISTANT] Here are the ingredients...",
        "[USER] Explain quantum mechanics in simple terms [ASSISTANT] Quantum mechanics is the study of...",
        "[USER] Hi Claude! How's your day going? [ASSISTANT] Hello! Thanks for asking, I'm doing well."
    ]
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for i, text in enumerate(test_samples, 1):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = ort_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()
        
        predicted_label = id_to_label.get(predicted_class_id, f"unknown_{predicted_class_id}")
        
        print(f"\n{i:2d}. Text: {text[:80]}...")
        print(f"    Prediction: {predicted_label} (confidence: {confidence:.3f})")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--model-path", default="./models/distilbert-intent-classifier", 
                       help="Path to trained PyTorch model")
    parser.add_argument("--output-path", default="./models/onnx-intent-classifier", 
                       help="Output path for ONNX model")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--no-optimize", action="store_true", help="Skip optimization")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after export")
    parser.add_argument("--test", action="store_true", help="Test inference after export")
    parser.add_argument("--benchmark-samples", type=int, default=100, 
                       help="Number of samples for benchmarking")
    
    args = parser.parse_args()
    
    try:
        # Export model
        onnx_path, config_info = export_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            quantize=not args.no_quantize,
            optimize=not args.no_optimize
        )
        
        # Run benchmark if requested
        if args.benchmark:
            benchmark_results = benchmark_model(str(onnx_path), args.benchmark_samples)
            
            # Save benchmark results
            with open(onnx_path / "benchmark_results.json", "w") as f:
                json.dump(benchmark_results, f, indent=2, default=str)
        
        # Test inference if requested
        if args.test:
            test_inference(str(onnx_path))
        
        print(f"\n✅ Export completed successfully!")
        print(f"ONNX model saved to: {onnx_path}")
        print(f"Quantized: {config_info['quantized']}")
        print(f"Optimized: {config_info['optimized']}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    main()