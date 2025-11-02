"""
Quick test of the exported ONNX model (FLOAT VERSION)
"""
import time
import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# model_path = "models/models_1024_base_bert/onnx"
model_path = "models/models_1024_base_distilbert/distilbert_distilled_1024/"
print(f"Loading FLOAT model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForSequenceClassification.from_pretrained(
    model_path,
    file_name="model.onnx"  # Specify the float model
)

print("Model loaded successfully!\n")

# Test conversation
test_text = "Hey Claude, can you help me debug this Python code? I'm getting a weird error with my list comprehension."

inputs = tokenizer(test_text, return_tensors="np", truncation=True, max_length=1024)

# Warm-up
_ = model(**inputs)

# Timed inference
times = []
for i in range(10):
    start = time.perf_counter()
    outputs = model(**inputs)
    end = time.perf_counter()
    times.append((end - start) * 1000)

logits = outputs.logits[0]
predicted_class = int(np.argmax(logits))

print(f"Test: {test_text[:80]}...")
print(f"\nPredicted class: {predicted_class}")
print(f"\nInference times:")
print(f"  Average: {np.mean(times):.2f}ms ✓" if np.mean(times) < 50 else f"  Average: {np.mean(times):.2f}ms ✗")
print(f"  Min: {np.min(times):.2f}ms")
print(f"  Median: {np.median(times):.2f}ms")

# Test with longer input
long_text = test_text * 20
inputs_long = tokenizer(long_text, return_tensors="np", truncation=True, max_length=1024)
print(f"\nLong input tokens: {inputs_long['input_ids'].shape[1]}")

start = time.perf_counter()
outputs_long = model(**inputs_long)
end = time.perf_counter()

print(f"Long input inference: {(end - start) * 1000:.2f}ms")