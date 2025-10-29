"""
Export a fine-tuned DistilBERT model to ONNX, quantize it, and run a test inference.

This script performs the following steps:
1.  Loads a trained DistilBERT model and tokenizer from a specified directory.
2.  Exports the model to the ONNX format, which allows for cross-platform inference.
3.  Applies dynamic quantization to the ONNX model to reduce its size and potentially speed up inference.
4.  Loads the quantized ONNX model using the ONNX Runtime.
5.  Runs a sample inference to verify that the quantized model works correctly and to show how to use it.
"""

import os
import json
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

def export_and_quantize_model(model_dir: str, output_dir: str):
    """
    Exports a trained DistilBERT model to ONNX, quantizes it, and saves the results.

    Args:
        model_dir (str): The directory where the fine-tuned model and tokenizer are saved.
        output_dir (str): The directory to save the ONNX and quantized ONNX models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the fine-tuned model and tokenizer
    print(f"Loading model and tokenizer from {model_dir}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()  # Set the model to evaluation mode

    # 2. Export the model to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting model to ONNX at {onnx_path}...")

    # Create a dummy input for tracing the model
    dummy_text = "This is a sample text for ONNX export."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )

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
    print("ONNX export complete.")

    # 3. Quantize the ONNX model
    quantized_onnx_path = os.path.join(output_dir, "model.quant.onnx")
    print(f"Applying dynamic quantization... Saving to {quantized_onnx_path}")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QInt8
    )
    print("Quantization complete.")

    # 4. Save tokenizer and training config for completeness
    tokenizer.save_pretrained(output_dir)
    training_config_path = os.path.join(model_dir, 'training_config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)

    print(f"Quantized model and tokenizer saved to {output_dir}")

def test_quantized_model(model_dir: str, text_to_classify: str):
    """
    Loads a quantized ONNX model and performs a test inference.

    Args:
        model_dir (str): The directory where the quantized model and tokenizer are saved.
        text_to_classify (str): A sample text string to classify.
    """
    print("\n" + "="*50)
    print("Running test inference with the quantized ONNX model...")

    # 1. Load the tokenizer and the class order from the config
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    class_order = config.get('class_order', [])

    # 2. Load the quantized ONNX model
    onnx_session = ort.InferenceSession(os.path.join(model_dir, "model.quant.onnx"))

    # 3. Prepare the input
    inputs = tokenizer(
        text_to_classify,
        return_tensors="np",
        max_length=512,
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
    predicted_class_label = class_order[predicted_class_index] if class_order else "N/A"
    
    print(f"\nText to classify: '{text_to_classify}'")
    print(f"Predicted class: {predicted_class_label} (index: {predicted_class_index})")
    print("Probabilities:")
    if class_order:
        for i, label in enumerate(class_order):
            print(f"  - {label}: {probabilities[0][i]:.4f}")
    else:
        print("  Class order not found in config.")


if __name__ == "__main__":
    # Define the directories
    # This is the directory where the trained model from train_distilBERT.py is saved
    TRAINED_MODEL_DIR = "models/distilbert_intent_classifier"
    # This is where the ONNX and quantized models will be saved
    ONNX_OUTPUT_DIR = "models/distilbert_intent_classifier_onnx"

    # Step 1: Export and quantize the model
    export_and_quantize_model(TRAINED_MODEL_DIR, ONNX_OUTPUT_DIR)

    # Step 2: Test the quantized model
    test_text = "This is a test sentence to see how the model classifies it."
    test_quantized_model(ONNX_OUTPUT_DIR, test_text)
