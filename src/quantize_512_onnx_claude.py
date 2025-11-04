#!/usr/bin/env python3
"""
Export PyTorch DistilBERT to Quantized ONNX (Workaround for broken quantize_dynamic)

Uses manual ONNX export with proper shape handling, then applies quantization
using a more robust method that doesn't rely on shape inference.
"""

import os
import json
import logging
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_and_quantize(model_dir: str, quantized_dir: str = None) -> str:
    """
    Export PyTorch model to quantized ONNX format
    """
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import onnx
        from onnx import numpy_helper
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        raise
    
    model_path = Path(model_dir)
    
    # Check for PyTorch model
    safetensors_file = model_path / "model.safetensors"
    if not safetensors_file.exists():
        raise FileNotFoundError(f"PyTorch model not found at {safetensors_file}")
    
    # Create output directory
    if quantized_dir is None:
        quantized_dir = model_path / "quantized"
    else:
        quantized_dir = Path(quantized_dir)
    
    quantized_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting ONNX export and quantization pipeline")
    logger.info(f"📂 Input model: {model_path}")
    logger.info(f"📂 Output directory: {quantized_dir}")
    
    # Step 1: Load PyTorch model
    logger.info("\n📦 Step 1: Loading PyTorch model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model loaded: {num_params:,} parameters")
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    num_labels = config.get('num_labels', 13)
    logger.info(f"🏷️  Number of labels: {num_labels}")
    
    # Step 2: Export to ONNX with proper settings
    logger.info("\n🔧 Step 2: Exporting to ONNX format...")
    
    onnx_path = quantized_dir / "model.onnx"
    
    # Create dummy input
    dummy_text = "This is a sample text for ONNX export."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    
    # Export with opset_version=11 to avoid LayerNormalization issues
    logger.info("📋 Using ONNX opset 11 for better compatibility...")
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask']),
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=11,  # Use opset 11 for stability
                do_constant_folding=True,
                export_params=True,
            )
        logger.info(f"✅ ONNX export successful: {onnx_path}")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        raise
    
    # Check model size
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"📊 ONNX model size: {onnx_size_mb:.2f} MB")
    
    if onnx_size_mb < 10:
        logger.error(f"❌ ONNX model suspiciously small ({onnx_size_mb:.2f} MB)!")
        logger.error("   This suggests the export didn't include all weights.")
        raise ValueError("ONNX export produced invalid small file")
    
    # Step 3: Try quantization with multiple methods
    logger.info("\n⚙️  Step 3: Attempting quantization...")
    
    quantized_successfully = False
    
    # Method 1: Try quantize_dynamic without shape inference
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
        from onnxruntime.quantization.registry import IntegerOpsRegistry
        
        logger.info("🔧 Method 1: Trying quantize_dynamic...")
        
        quantized_path = quantized_dir / "model_quantized.onnx"
        
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QUInt8,
        )
        
        # Replace original with quantized
        if quantized_path.exists():
            onnx_path.unlink()
            quantized_path.rename(onnx_path)
            
            quantized_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            reduction = (1 - (quantized_size_mb / onnx_size_mb)) * 100
            
            logger.info(f"✅ Quantization successful!")
            logger.info(f"📊 Quantized size: {quantized_size_mb:.2f} MB ({reduction:.1f}% reduction)")
            quantized_successfully = True
            
    except Exception as e:
        logger.warning(f"⚠️  Method 1 failed: {e}")
    
    # Method 2: Use onnxmltools if Method 1 failed
    if not quantized_successfully:
        try:
            import onnxmltools
            from onnxmltools.utils.float16_converter import convert_float_to_float16
            
            logger.info("🔧 Method 2: Trying onnxmltools FP16 conversion...")
            
            # Load model
            model_onnx = onnx.load(str(onnx_path))
            
            # Convert to FP16 (lighter weight, still good accuracy)
            model_fp16 = convert_float_to_float16(model_onnx)
            
            # Save
            fp16_path = quantized_dir / "model_fp16.onnx"
            onnx.save(model_fp16, str(fp16_path))
            
            # Replace original
            onnx_path.unlink()
            fp16_path.rename(onnx_path)
            
            quantized_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            reduction = (1 - (quantized_size_mb / onnx_size_mb)) * 100
            
            logger.info(f"✅ FP16 conversion successful!")
            logger.info(f"📊 FP16 model size: {quantized_size_mb:.2f} MB ({reduction:.1f}% reduction)")
            quantized_successfully = True
            
        except ImportError:
            logger.warning("⚠️  onnxmltools not installed, skipping FP16 method")
            logger.warning("   Install with: pip install onnxmltools")
        except Exception as e:
            logger.warning(f"⚠️  Method 2 failed: {e}")
    
    if not quantized_successfully:
        logger.warning("⚠️  All quantization methods failed")
        logger.warning("   Using unquantized ONNX model")
        logger.info(f"📊 Final model size: {onnx_size_mb:.2f} MB (unquantized)")
    
    # Step 4: Copy supporting files
    logger.info("\n📋 Step 4: Copying supporting files...")
    
    supporting_files = [
        "config.json",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
    ]
    
    custom_files = [
        "onnx_config.json",
        "training_config.json",
        "training_completion_summary.json"
    ]
    
    copied_count = 0
    for file in supporting_files + custom_files:
        src = model_path / file
        if src.exists():
            dst = quantized_dir / file
            shutil.copy2(src, dst)
            copied_count += 1
            logger.info(f"  ✓ Copied {file}")
    
    logger.info(f"📋 Copied {copied_count} supporting files")
    
    # Create ONNX config
    onnx_config = {
        "max_length": 512,
        "class_order": get_class_order_from_config(model_path),
        "num_labels": num_labels
    }
    
    onnx_config_path = quantized_dir / "onnx_config.json"
    with open(onnx_config_path, 'w') as f:
        json.dump(onnx_config, f, indent=2)
    
    logger.info(f"💾 Created onnx_config.json")
    
    # Save export metadata
    final_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    
    export_info = {
        "original_pytorch_params": num_params,
        "onnx_model_size_mb": round(final_size_mb, 2),
        "quantized": quantized_successfully,
        "quantization_method": "dynamic_uint8" if quantized_successfully else "none",
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    info_path = quantized_dir / "export_info.json"
    with open(info_path, 'w') as f:
        json.dump(export_info, f, indent=2)
    
    logger.info(f"\n✅ Export complete!")
    
    return str(quantized_dir)

def get_class_order_from_config(model_path: Path) -> List[str]:
    """Extract class order from model config or training files"""
    
    # Try training_completion_summary.json first
    summary_path = model_path / "training_completion_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            if 'class_order' in summary:
                return summary['class_order']
    
    # Try training_config.json
    training_config_path = model_path / "training_config.json"
    if training_config_path.exists():
        with open(training_config_path) as f:
            training_config = json.load(f)
            if 'class_order' in training_config:
                return training_config['class_order']
    
    # Try model config.json id2label
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if 'id2label' in config:
                id2label = config['id2label']
                return [id2label[str(i)] for i in sorted(int(k) for k in id2label.keys())]
    
    logger.warning("⚠️  Could not find class_order")
    return []

def test_onnx_model(quantized_dir: str, test_texts: List[str] = None) -> Dict:
    """Test the ONNX model with sample inputs"""
    
    try:
        import onnxruntime as ort
        from transformers import DistilBertTokenizer
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        raise
    
    quantized_path = Path(quantized_dir)
    model_file = quantized_path / "model.onnx"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found at {model_file}")
    
    logger.info(f"\n🧪 Testing ONNX model...")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(str(quantized_path))
    
    # Load config
    onnx_config_path = quantized_path / "onnx_config.json"
    if onnx_config_path.exists():
        with open(onnx_config_path) as f:
            onnx_config = json.load(f)
            max_length = onnx_config.get('max_length', 512)
            class_order = onnx_config.get('class_order', [])
    else:
        max_length = 512
        class_order = []
    
    logger.info(f"📏 Max length: {max_length}")
    logger.info(f"🏷️  Classes ({len(class_order)}): {class_order}")
    
    # Create ONNX session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        str(model_file),
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    
    logger.info(f"✅ Model loaded successfully")
    
    # Default test texts
    if test_texts is None:
        test_texts = [
            "Can you help me with my homework on calculus?",
            "I need coding help with Python decorators",
            "Write me a story about dragons and knights",
            "I'm feeling really sad and anxious today",
            "What's the weather like in San Francisco?",
            "How do I make authentic Italian pasta?",
            "Generate marketing ideas for my startup",
            "Translate this sentence to Spanish please",
        ]
    
    results = []
    total_time = 0
    
    logger.info(f"\n🔍 Running {len(test_texts)} test predictions...")
    logger.info("="*70)
    
    for i, text in enumerate(test_texts, 1):
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
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
        total_time += inference_time
        
        logits = outputs[0]
        predicted_class_idx = np.argmax(logits, axis=1)[0]
        confidence = np.max(softmax(logits[0]))
        
        predicted_class = class_order[predicted_class_idx] if class_order else f"Class_{predicted_class_idx}"
        
        result = {
            'text': text,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'inference_time_ms': round(inference_time, 2)
        }
        results.append(result)
        
        text_preview = text[:60] + "..." if len(text) > 60 else text
        logger.info(f"  Test {i}: '{text_preview}'")
        logger.info(f"    → {predicted_class} (conf: {confidence:.3f}, {inference_time:.2f}ms)")
    
    logger.info("\n" + "="*70)
    
    avg_time = total_time / len(test_texts)
    min_time = min(r['inference_time_ms'] for r in results)
    max_time = max(r['inference_time_ms'] for r in results)
    
    logger.info(f"📊 Performance Summary:")
    logger.info(f"  Average: {avg_time:.2f}ms | Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")
    
    if avg_time <= 30:
        logger.info(f"  🎯 Performance: EXCELLENT (≤30ms)")
    elif avg_time <= 50:
        logger.info(f"  ✅ Performance: GOOD (≤50ms)")
    else:
        logger.info(f"  ⚠️  Performance: ACCEPTABLE but >50ms")
    
    test_results = {
        'num_tests': len(test_texts),
        'avg_inference_time_ms': round(avg_time, 2),
        'min_inference_time_ms': round(min_time, 2),
        'max_inference_time_ms': round(max_time, 2),
        'predictions': results,
        'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_path = quantized_path / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"\n✅ Test results saved to {results_path}")
    
    return test_results

def softmax(x):
    """Compute softmax values"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Export PyTorch to ONNX")
    parser.add_argument("--model-dir", default="models/distilbert_distilled_alpha_0.0",
                       help="Directory containing PyTorch model")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: model_dir/quantized)")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip testing after export")
    
    args = parser.parse_args()
    
    logger.info("🚀 PyTorch → ONNX Export Pipeline")
    logger.info("="*70)
    
    try:
        # Export
        quantized_dir = export_and_quantize(args.model_dir, args.output_dir)
        
        # Test if requested
        if not args.skip_test:
            test_results = test_onnx_model(quantized_dir)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"📂 Model location: {quantized_dir}")
        if not args.skip_test:
            logger.info(f"⚡ Average inference: {test_results['avg_inference_time_ms']:.2f}ms")
        logger.info("🎯 Model ready for production!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())