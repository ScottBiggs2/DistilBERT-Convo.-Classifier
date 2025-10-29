
import os
import json
import time
import torch
import onnxruntime as ort
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import glob

def load_original_model(model_dir: str):
    """Loads the original fine-tuned DistilBERT model and tokenizer."""
    print(f"Loading original model and tokenizer from {model_dir}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def load_onnx_model(model_dir: str):
    """Loads the quantized ONNX model and tokenizer."""
    print(f"Loading ONNX model and tokenizer from {model_dir}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    session = ort.InferenceSession(os.path.join(model_dir, "model.quant.onnx"))
    return tokenizer, session

def predict_original(model, tokenizer, text, class_order):
    """Performs inference with the original model."""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()
    predicted_class_index = np.argmax(probabilities, axis=-1)[0]
    return class_order[predicted_class_index]

def predict_onnx(session, tokenizer, text, class_order):
    """Performs inference with the ONNX model."""
    inputs = tokenizer(
        text,
        return_tensors="np",
        max_length=512,
        padding='max_length',
        truncation=True
    )
    input_feed = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    outputs = session.run(None, input_feed)
    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    predicted_class_index = np.argmax(probabilities, axis=-1)[0]
    return class_order[predicted_class_index]

def compare_models(original_model_dir: str, onnx_model_dir: str, data_dir: str):
    """Compares the original and ONNX models on the given data."""
    # Load models
    original_tokenizer, original_model = load_original_model(original_model_dir)
    onnx_tokenizer, onnx_session = load_onnx_model(onnx_model_dir)

    # Load class order from the original model's config
    config_path = os.path.join(original_model_dir, 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    class_order = config.get('class_order', [])
    if not class_order:
        print("Class order not found in config. Exiting.")
        return

    # Find data files
    data_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not data_files:
        print(f"No JSON files found in {data_dir}. Exiting.")
        return

    # Initialize comparison metrics
    total_samples = 0
    matches = 0
    original_total_time = 0
    onnx_total_time = 0

    # Iterate through data files
    for data_file in data_files:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            text = item.get("text")
            true_label = item.get("label")

            if not text or not true_label:
                continue

            total_samples += 1

            # Original model inference
            start_time = time.time()
            original_pred = predict_original(original_model, original_tokenizer, text, class_order)
            original_total_time += time.time() - start_time

            # ONNX model inference
            start_time = time.time()
            onnx_pred = predict_onnx(onnx_session, onnx_tokenizer, text, class_order)
            onnx_total_time += time.time() - start_time

            if original_pred == onnx_pred:
                matches += 1

    # Print results
    print("\n" + "="*50)
    print("Model Comparison Results")
    print("="*50)
    print(f"Total samples: {total_samples}")
    print(f"Predictions matched: {matches} ({matches/total_samples*100:.2f}%)")
    print(f"Original model average inference time: {original_total_time/total_samples:.6f} seconds")
    print(f"ONNX model average inference time: {onnx_total_time/total_samples:.6f} seconds")


if __name__ == "__main__":
    ORIGINAL_MODEL_DIR = "models/distilbert_intent_classifier"
    ONNX_MODEL_DIR = "models/distilbert_intent_classifier_onnx"
    DATA_DIR = "data/eval"
    
    compare_models(ORIGINAL_MODEL_DIR, ONNX_MODEL_DIR, DATA_DIR)
