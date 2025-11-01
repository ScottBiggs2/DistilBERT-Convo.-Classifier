"""
Export a fine-tuned Transformer model to ONNX, quantize it, and run a test inference.

This script performs the following steps:
1.  Loads a trained model and tokenizer from a specified directory using AutoModel.
2.  Exports the model to the ONNX format.
3.  Applies dynamic quantization to the ONNX model.
4.  Runs a sample inference to verify the quantized model.
"""

import os
import json
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import argparse

def export_and_quantize_model(model_dir: str, output_dir: str, max_length: int):
    """
    Exports a trained Transformer model to ONNX, quantizes it, and saves the results.

    Args:
        model_dir (str): The directory where the fine-tuned model and tokenizer are saved.
        output_dir (str): The directory to save the ONNX and quantized ONNX models.
        max_length (int): The maximum sequence length for the model input.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the fine-tuned model and tokenizer using Auto-classes
    print(f"Loading model and tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # Set the model to evaluation mode

    # 2. Export the model to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting model to ONNX at {onnx_path} with max_length={max_length}...")

    # Create a dummy input for tracing the model
    dummy_text = "This is a sample text for ONNX export."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    # Wrap the model so we export only the logits tensor and have a stable forward signature
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = _ExportWrapper(model)
    wrapper.eval()

    input_names = ['input_ids', 'attention_mask']
    output_names = ['logits']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'attention_mask': {0: 'batch_size', 1: 'seq_len'},
        'logits': {0: 'batch_size'}
    }

    # Export with a clean forward signature
    try:
        torch.onnx.export(
            wrapper,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            do_constant_folding=True
        )
        print("ONNX export complete.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        raise

    # Validate the ONNX model and run shape inference before attempting quantization
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        try:
            inferred = onnx.shape_inference.infer_shapes(onnx_model)
            # optionally overwrite the saved model with inferred shapes for downstream tools
            onnx.save(inferred, onnx_path)
            print("ONNX model shape inference succeeded and model saved with inferred shapes.")
        except Exception as si_e:
            print(f"ONNX shape inference failed: {si_e}. Continuing, but quantization may fail.")
    except Exception as ck_e:
        print(f"ONNX checker failed: {ck_e}. Aborting quantization step.")

    # 3. Quantize the ONNX model (if desired)
    quantized_onnx_path = os.path.join(output_dir, "model.quant.onnx")
    try:
        print(f"Applying dynamic quantization... Saving to {quantized_onnx_path}")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_onnx_path,
            weight_type=QuantType.QInt8
        )
        print("Quantization complete.")
    except Exception as q_e:
        print(f"Quantization failed: {q_e}")
        print("Skipping saving quantized model. The float ONNX model is available at {onnx_path}.")

    # 4. Save tokenizer and training config for completeness
    tokenizer.save_pretrained(output_dir)
    training_config_path = os.path.join(model_dir, 'training_config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)

    print(f"Quantized model and tokenizer saved to {output_dir}")


def test_quantized_model(model_dir: str, text_to_classify: str, max_length: int):
    """
    Loads a quantized ONNX model and performs a test inference.

    Args:
        model_dir (str): The directory where the quantized model and tokenizer are saved.
        text_to_classify (str): A sample text string to classify.
        max_length (int): The maximum sequence length for the model input.
    """
    print("\n" + "="*50)
    print("Running test inference with the quantized ONNX model...")

    # 1. Load the tokenizer and the class order from the config
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config_path = os.path.join(model_dir, 'training_config.json')
    class_order = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        class_order = config.get('class_order', [])
    else:
        print("Warning: training_config.json not found. Class labels will not be available.")


    # 2. Load the quantized ONNX model (fall back to float ONNX if quant not present)
    quant_path = os.path.join(model_dir, "model.quant.onnx")
    float_path = os.path.join(model_dir, "model.onnx")
    chosen = quant_path if os.path.exists(quant_path) else float_path
    if not os.path.exists(chosen):
        raise FileNotFoundError(f"No ONNX model found in {model_dir} (checked quant and float paths)")
    onnx_session = ort.InferenceSession(chosen)

    # 3. Prepare the input
    inputs = tokenizer(
        text_to_classify,
        return_tensors="np",
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    input_feed = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }

    # 4. Run inference
    outputs = onnx_session.run(None, input_feed)
    logits = outputs[0]

    # 5. Process the output
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predicted_class_index = np.argmax(probabilities, axis=-1)[0]
    
    predicted_class_label = "N/A"
    if class_order:
        if predicted_class_index < len(class_order):
            predicted_class_label = class_order[predicted_class_index]
        else:
            print(f"Warning: Predicted class index {predicted_class_index} is out of bounds for class_order list of size {len(class_order)}.")

    print(f"\nText to classify: '{text_to_classify}'")
    print(f"Predicted class: {predicted_class_label} (index: {predicted_class_index})")
    print("Probabilities:")
    if class_order:
        for i, label in enumerate(class_order):
            print(f"  - {label}: {probabilities[0][i]:.4f}")
    else:
        print("  Class order not found in config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Transformer model to ONNX and quantize it.")
    parser.add_argument("--model-path", type=str, required=True, help="Directory of the fine-tuned model.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the ONNX models.")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length for the model.")
    parser.add_argument("--test", action="store_true", help="Run a test inference after exporting.")
    
    args = parser.parse_args()

    # Step 1: Export and quantize the model
    export_and_quantize_model(args.model_path, args.output_path, args.max_length)

    # Step 2: Test the quantized model if requested
    if args.test:
        test_text = "This is a test sentence to see how the model classifies it."
        test_quantized_model(args.output_path, test_text, args.max_length)
