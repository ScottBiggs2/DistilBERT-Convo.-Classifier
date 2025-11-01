#!/usr/bin/env python3
"""
Model Evaluation Framework for Heuristic Comparison

Compares the behavior of multiple models on a given dataset without ground truth.

Models to be evaluated:
1. DistilBERT (1024) - ONNX
2. DistilBERT (1024) - Full Transformers
3. BERT (1024) - ONNX
4. BERT (1024) - Full Transformers
5. Llama 3.1 8B via Groq
6. Gemini 2.5 Flash Lite via Google GenAI

Metrics:
- Label Distribution
- Inter-Model Agreement Matrix
- Inference Speed
- Side-by-Side Comparison Output
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from groq import Groq
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import sys

# --- Configuration ---
load_dotenv()

# Avoid tokenizers parallelism / fork warnings and potential deadlocks in multiprocessing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

# Ensure project root is on sys.path so sibling packages (like `src`) can be imported
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.label_prompts import LETTER_LABEL_PROMPT, INTENT_CATEGORIES_LIST, EXAMPLES_LIST

DATASET_PATH = os.path.join(SCRIPT_DIR, "evaluation_dataset.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODELS_TO_EVALUATE = {
    "distilbert_onnx": {
        "type": "onnx",
        "path": os.path.join(PROJECT_ROOT, "models", "models_1024_base_distilbert", "distilbert_distilled_1024"),
        "onnx_file": "model.onnx",
        "max_length": 1024
    },
    "distilbert_onnx_quant": {
        "type": "onnx",
        "path": os.path.join(PROJECT_ROOT, "models", "models_1024_base_distilbert", "distilbert_distilled_1024"),
        "onnx_file": "model.quant.onnx",
        "max_length": 1024
    },
    "distilbert_full": {
        "type": "transformers",
        "path": os.path.join(PROJECT_ROOT, "models", "models_1024_base_distilbert", "distilbert_distilled_1024"),
        "max_length": 1024
    },
    "bert_onnx": {
        "type": "onnx",
        "path": os.path.join(PROJECT_ROOT, "models", "models_1024_base_bert", "bert_distilled_1024_onnx"),
        "onnx_file": "model.onnx",
        "max_length": 1024
    },
    "bert_onnx_quant": {
        "type": "onnx",
        "path": os.path.join(PROJECT_ROOT, "models", "models_1024_base_bert", "bert_distilled_1024_onnx"),
        "onnx_file": "model.quant.onnx",
        "max_length": 1024
    },
    "bert_full": {
        "type": "transformers",
        "path": os.path.join(PROJECT_ROOT, "models", "models_1024_base_bert", "bert_distilled_1024"),
        "max_length": 1024
    },
    "llama3.1_8b_groq": {
        "type": "groq",
        "model_name": "llama3.1-8b-instant"
    },
    "gemini_2.5_flash": {
        "type": "gemini",
        "model_name": "gemini-2.5-flash-lite"
    }
}

# --- Model Interface Functions ---

_model_cache = {}

def run_onnx_inference(model_path, text, max_length):
    # Accepts model_path as a dict with 'path' and 'onnx_file' keys
    if isinstance(model_path, dict):
        base_path = model_path["path"]
        onnx_file = model_path["onnx_file"]
    else:
        base_path = model_path
        onnx_file = "model.onnx"
    # Resolve ONNX file: prefer the requested file, but fall back to common names when missing
    requested_path = os.path.join(base_path, onnx_file)
    chosen_onnx = None

    if os.path.exists(requested_path):
        chosen_onnx = requested_path
    else:
        # try common alternatives
        alt_candidates = [os.path.join(base_path, "model.quant.onnx"), os.path.join(base_path, "model.onnx")]
        for cand in alt_candidates:
            if os.path.exists(cand):
                chosen_onnx = cand
                break

    if chosen_onnx is None:
        # fallback: pick the first .onnx file in the directory if any
        try:
            import glob
            onnx_list = glob.glob(os.path.join(base_path, "*.onnx"))
            if onnx_list:
                chosen_onnx = onnx_list[0]
        except Exception:
            chosen_onnx = None

    if chosen_onnx is None:
        raise FileNotFoundError(f"No ONNX model found in {base_path} (tried {requested_path} and alternatives)")

    cache_key = f"{base_path}:{os.path.basename(chosen_onnx)}"
    if cache_key not in _model_cache:
        print(f"Loading ONNX model from {chosen_onnx}...")
        # Configure ONNXRuntime session options for CPU
        sess_options = ort.SessionOptions()
        try:
            sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 1))
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass

        session = ort.InferenceSession(chosen_onnx, sess_options)
        tokenizer_obj = AutoTokenizer.from_pretrained(base_path)
        _model_cache[cache_key] = {
            "session": session,
            "tokenizer": tokenizer_obj
        }
        # load class order if available
        config_path = os.path.join(base_path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            _model_cache[cache_key]["class_order"] = config.get('class_order', [])
        else:
            _model_cache[cache_key]["class_order"] = [chr(ord('A') + i) for i in range(13)]

        # Log providers and warm up session (short)
        try:
            providers = session.get_providers()
            print(f"ONNXRuntime providers for {os.path.basename(chosen_onnx)}: {providers}")
        except Exception:
            pass
        try:
            warmup_inputs = tokenizer_obj("warmup", return_tensors="np", max_length=16, padding='max_length', truncation=True)
            for _ in range(3):
                session.run(None, {'input_ids': warmup_inputs['input_ids'], 'attention_mask': warmup_inputs['attention_mask']})
            print(f"Warmup complete for {os.path.basename(chosen_onnx)}")
        except Exception as e:
            print(f"Warmup failed for {os.path.basename(chosen_onnx)}: {e}")

    session = _model_cache[cache_key]["session"]
    tokenizer = _model_cache[cache_key]["tokenizer"]
    class_order = _model_cache[cache_key]["class_order"]

    # Separate tokenization and model timing to diagnose slowdowns
    start_time = time.time()
    t_token_start = time.time()
    inputs = tokenizer(text, return_tensors="np", max_length=max_length, padding='max_length', truncation=True)
    token_time = time.time() - t_token_start

    t_model_start = time.time()
    input_feed = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
    outputs = session.run(None, input_feed)
    model_time = time.time() - t_model_start

    logits = outputs[0]
    # Use numpy softmax to avoid torch <-> numpy conversions
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    predicted_class_index = np.argmax(probabilities, axis=-1)[0]
    inference_time = time.time() - start_time
    prediction = class_order[predicted_class_index] if class_order and predicted_class_index < len(class_order) else "N/A"
    return {"prediction": prediction, "inference_time": inference_time, "token_time": token_time, "model_time": model_time}

def run_transformers_inference(model_path, text, max_length):
    if model_path not in _model_cache:
        print(f"Loading Transformers model from {model_path}...")
        # Load model and tokenizer, then do a small warmup to avoid first-call overhead
        model_obj = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer_obj = AutoTokenizer.from_pretrained(model_path)
        _model_cache[model_path] = {
            "model": model_obj,
            "tokenizer": tokenizer_obj
        }
        config_path = os.path.join(model_path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            _model_cache[model_path]["class_order"] = config.get('class_order', [])
        else:
            _model_cache[model_path]["class_order"] = [chr(ord('A') + i) for i in range(13)]
        # Try a short warmup to reduce first-inference latency
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 1)))
            warmup_inputs = tokenizer_obj("warmup", return_tensors="pt", max_length=16, padding='max_length', truncation=True)
            model_obj.eval()
            with torch.no_grad():
                for _ in range(2):
                    _ = model_obj(**{k: v.to(model_obj.device) for k, v in warmup_inputs.items()})
            print(f"Warmup complete for transformers model at {model_path}")
        except Exception as e:
            print(f"Transformers warmup failed for {model_path}: {e}")

    model = _model_cache[model_path]["model"]
    tokenizer = _model_cache[model_path]["tokenizer"]
    class_order = _model_cache[model_path]["class_order"]

    # Split tokenization and model timing for diagnostics
    start_time = time.time()
    t_token_start = time.time()
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True)
    token_time = time.time() - t_token_start

    t_model_start = time.time()
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
    model_time = time.time() - t_model_start

    logits = outputs.logits
    # Move logits to CPU numpy and apply numpy softmax
    logits_np = logits.detach().cpu().numpy()
    exp_logits = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    predicted_class_index = np.argmax(probabilities, axis=-1)[0]
    inference_time = time.time() - start_time
    prediction = class_order[predicted_class_index] if class_order and predicted_class_index < len(class_order) else "N/A"
    return {"prediction": prediction, "inference_time": inference_time, "token_time": token_time, "model_time": model_time}

def run_groq_inference(model_name, text):
    if "groq_client" not in _model_cache:
        print("Initializing Groq client...")
        _model_cache["groq_client"] = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    client = _model_cache["groq_client"]
    start_time = time.time()
    # Use the shared labeling prompt (single-letter output) so Groq and Gemini classify the same task
    prompt = LETTER_LABEL_PROMPT.format(
        intent_categories_list=INTENT_CATEGORIES_LIST,
        examples_list=EXAMPLES_LIST,
        conversation_text=text
    )

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
    )
    prediction = chat_completion.choices[0].message.content.strip()
    inference_time = time.time() - start_time
    return {"prediction": prediction, "inference_time": inference_time}

def run_gemini_inference(model_name, text):
    if "gemini_model" not in _model_cache:
        print("Initializing Gemini client...")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _model_cache["gemini_model"] = genai.GenerativeModel(model_name)

    model = _model_cache["gemini_model"]
    start_time = time.time()
    # Use the same single-letter prompt used elsewhere to ensure consistent labeling
    prompt = LETTER_LABEL_PROMPT.format(
        intent_categories_list=INTENT_CATEGORIES_LIST,
        examples_list=EXAMPLES_LIST,
        conversation_text=text
    )

    response = model.generate_content(prompt)
    prediction = response.text.strip()
    inference_time = time.time() - start_time
    return {"prediction": prediction, "inference_time": inference_time}

# --- Results Analysis ---

def analyze_and_save_results(results_df):
    print("\nEvaluation complete. Analyzing results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report = "# Model Comparison Summary\n\n"

    # --- Inference Speed ---
    def safe_mean_ms(col_name):
        if col_name in results_df.columns:
            return results_df[col_name].dropna().mean() * 1000
        return float('nan')

    avg_inference_times = {model: safe_mean_ms(f"{model}_time") for model in MODELS_TO_EVALUATE}
    report += "## Average Inference Speed\n\n| Model | Avg. Inference Time (ms) |\n|---|---|"
    for model, speed in avg_inference_times.items():
        if np.isnan(speed):
            report += f"| {model} | N/A |\n"
        else:
            report += f"| {model} | {speed:.2f} |\n"
    report += "\n"

    # --- Tokenization/Model time breakdown ---
    report += "## Average Tokenization and Model Time (ms)\n\n| Model | Tokenization (ms) | Model (ms) |\n|---|---:|---:|\n"
    avg_token_times = {}
    avg_model_times = {}
    for model in MODELS_TO_EVALUATE:
        t_ms = safe_mean_ms(f"{model}_token_time")
        m_ms = safe_mean_ms(f"{model}_model_time")
        avg_token_times[model] = t_ms
        avg_model_times[model] = m_ms
        t_str = "N/A" if np.isnan(t_ms) else f"{t_ms:.2f}"
        m_str = "N/A" if np.isnan(m_ms) else f"{m_ms:.2f}"
        report += f"| {model} | {t_str} | {m_str} |\n"
    report += "\n"

    # Create token/model time charts (separate charts)
    try:
        token_vals = [avg_token_times[m] if not np.isnan(avg_token_times[m]) else 0 for m in MODELS_TO_EVALUATE]
        model_vals = [avg_model_times[m] if not np.isnan(avg_model_times[m]) else 0 for m in MODELS_TO_EVALUATE]
        labels = list(MODELS_TO_EVALUATE.keys())

        plt.figure(figsize=(12, 6))
        sns.barplot(x=labels, y=token_vals)
        plt.title('Average Tokenization Time (ms)')
        plt.ylabel('Tokenization (ms)')
        plt.xlabel('Model')
        token_chart_path = os.path.join(RESULTS_DIR, 'avg_token_time.png')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(token_chart_path)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=labels, y=model_vals)
        plt.title('Average Model Inference Time (ms)')
        plt.ylabel('Model (ms)')
        plt.xlabel('Model')
        model_chart_path = os.path.join(RESULTS_DIR, 'avg_model_time.png')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(model_chart_path)
        plt.close()

        report += f"![Avg tokenization time]({os.path.basename(token_chart_path)})\n\n"
        report += f"![Avg model time]({os.path.basename(model_chart_path)})\n\n"
    except Exception as e:
        report += f"\nCould not generate timing charts: {e}\n\n"

    # --- Label Distribution ---
    report += "## Label Distribution per Model\n\n"
    for model_name in MODELS_TO_EVALUATE:
        report += f"### {model_name}\n\n"
        dist = results_df[f"{model_name}_pred"].value_counts().sort_index()
        report += dist.to_markdown() + "\n\n"

        plt.figure(figsize=(10, 6))
        sns.barplot(x=dist.index, y=dist.values)
        plt.title(f"Label Distribution - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("Count")
        dist_chart_path = os.path.join(RESULTS_DIR, f"{model_name}_distribution.png")
        plt.savefig(dist_chart_path)
        plt.close()
        report += f"![Label Distribution for {model_name}]({os.path.basename(dist_chart_path)})\n\n"

    # --- Inter-Model Agreement ---
    report += "## Inter-Model Agreement Matrix\n\nPercentage of times models agreed on the prediction.\n\n"
    model_preds = [col for col in results_df.columns if col.endswith("_pred")]
    agreement_df = pd.DataFrame(index=MODELS_TO_EVALUATE.keys(), columns=MODELS_TO_EVALUATE.keys(), dtype=float)

    for model1 in MODELS_TO_EVALUATE:
        for model2 in MODELS_TO_EVALUATE:
            agreement = np.mean(results_df[f"{model1}_pred"] == results_df[f"{model2}_pred"])
            agreement_df.loc[model1, model2] = agreement

    report += agreement_df.to_markdown(floatfmt=".2%") + "\n\n"

    # --- Save Side-by-Side CSV ---
    csv_path = os.path.join(RESULTS_DIR, "side_by_side_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    report += f"A detailed side-by-side comparison has been saved to [{os.path.basename(csv_path)}]({os.path.basename(csv_path)}).\n"

    # --- Save Final Report ---
    report_path = os.path.join(RESULTS_DIR, "evaluation_results.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Results saved to {RESULTS_DIR}")

# --- Evaluation Orchestrator ---

def run_evaluation():
    """Orchestrates the model evaluation process."""
    print("Starting model evaluation...")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Evaluation dataset not found at {DATASET_PATH}")
        return
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} records for evaluation.")

    all_results = []

    for item in tqdm(dataset, desc="Evaluating models"):
        text = item.get("text_for_model", "")
        if not text:
            continue

        record = {"chat_id": item["chat_id"], "text_for_model": text}

        for model_name, model_config in MODELS_TO_EVALUATE.items():
            try:
                if model_config["type"] == "onnx":
                    # pass the whole config so run_onnx_inference can pick the correct onnx_file
                    result = run_onnx_inference(model_config, text, model_config["max_length"])
                elif model_config["type"] == "transformers":
                    result = run_transformers_inference(model_config["path"], text, model_config["max_length"])
                elif model_config["type"] == "groq":
                    result = run_groq_inference(model_config["model_name"], text)
                elif model_config["type"] == "gemini":
                    result = run_gemini_inference(model_config["model_name"], text)
                
                record[f"{model_name}_pred"] = result.get("prediction", "N/A")
                record[f"{model_name}_time"] = result.get("inference_time", 0)
                # Persist tokenization and model timing when provided by runtimes
                if "token_time" in result:
                    record[f"{model_name}_token_time"] = result.get("token_time", None)
                else:
                    record[f"{model_name}_token_time"] = None
                if "model_time" in result:
                    record[f"{model_name}_model_time"] = result.get("model_time", None)
                else:
                    record[f"{model_name}_model_time"] = None

            except Exception as e:
                print(f"Error processing model {model_name} for item {item['chat_id']}: {e}")
                record[f"{model_name}_pred"] = "ERROR"
                record[f"{model_name}_time"] = 0
        
        all_results.append(record)

    results_df = pd.DataFrame(all_results)
    analyze_and_save_results(results_df)

if __name__ == "__main__":
    run_evaluation()