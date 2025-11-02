"""
Quick test of the exported ONNX model
"""
import time
import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Load the quantized model
model_path = "models/models_1024_base_bert/onnx"
print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ORTModelForSequenceClassification.from_pretrained(
    model_path,
    file_name="model.quant.onnx"  # Use the quantized version
)

print("Model loaded successfully!\n")

# Test conversation
test_text = "Hey Claude, can you help me debug this Python code? I'm getting a weird error with my list comprehension."

# Tokenize
inputs = tokenizer(test_text, return_tensors="np", truncation=True, max_length=1024)

# Warm-up run
_ = model(**inputs)

# Timed inference
times = []
for i in range(10):
    start = time.perf_counter()
    outputs = model(**inputs)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # Convert to ms

# Get prediction
logits = outputs.logits[0]
predicted_class = int(np.argmax(logits))
confidence = float(np.max(logits))

print(f"Test input: {test_text[:100]}...")
print(f"\nPredicted class: {predicted_class}")
print(f"Confidence: {confidence:.3f}")
print(f"\nInference times (ms):")
print(f"  Average: {np.mean(times):.2f}ms")
print(f"  Min: {np.min(times):.2f}ms")
print(f"  Max: {np.max(times):.2f}ms")
print(f"  Median: {np.median(times):.2f}ms")

# Test with longer input
long_text = test_text * 20  # Repeat to test 1024 token handling
inputs_long = tokenizer(long_text, return_tensors="np", truncation=True, max_length=1024)
print(f"\nLong input tokens: {inputs_long['input_ids'].shape[1]}")

start = time.perf_counter()
outputs_long = model(**inputs_long)
end = time.perf_counter()

print(f"Long input inference time: {(end - start) * 1000:.2f}ms")