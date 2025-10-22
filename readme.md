*** NOT CURRENT TREE ***
topic-classifier/
├─ data/
│ ├─ records.json # raw JSON files [DONE]
│ ├─ raw_sequences.json # JSON with extracted message sequences [DONE]
│ ├─ cleaned_sequences.json # Cleaned conversation sequences [DONE]
│ ├─ gpt4o_labelled.json # full labeled dataset [DONE]
│ └─ verified_gpt5.json # small sample for dev [DONE]
├─ models/
│ ├─ distilbert-intent-classifier/ # PyTorch model from training [DONE]
│ │ └─ checkpoint-XX/ # HF model checkpoints [DONE]
│ └─ onnx-intent-classifier/ # NEW: Quantized ONNX model for serving
│    ├─ model.onnx
│    ├─ tokenizer_config.json
│    ├─ vocab.txt
│    ├─ model_config.json
│    └─ benchmark_results.json
├─ src/
│ ├─ extract_sequences.py # [DONE]
│ ├─ preprocess_sequences.py # [DONE] 
│ ├─ label_and_audit.py # [DONE]
│ ├─ train.py # [UPDATED]
│ ├─ export_onnx.py # [NEW]
│ ├─ serve_fastapi.py # [NEW]
│ ├─ utils.py # [NEW]
│ └─ rules.py # [DONE]
├─ deployment/ # NEW: Docker and deployment files
│ ├─ Dockerfile
│ ├─ docker-compose.yml
│ ├─ deploy.sh
│ └─ test_api.py
├─ requirements.txt # [UPDATED with ONNX/FastAPI deps]
├─ README.md # [UPDATE with deployment instructions]
└─ experiments/
  ├─ config_distilbert.yaml
  └─ notes.md


# Message Cleaning: 

Beginning with a ```records.json``` in the data folder. Example of structure: 
```json    
    "chat_id": "sample",
    "context": "sample",
    "query": "sample",
    "messages": [
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "user",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "assistant",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "user",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "assistant",
        "content": "stuff"
      }
    ],
    "formatted_chat": "User: stuff"
```

*** TODO: Go make sure that long conversation histories are properly dealt with ***
```bash 
python src/extract_sequences.py
```

```bash 
python src/preprocess_sequences.py
```

# Label Message Sequences and Audit Label Quality:

```bash
python src/label_and_audit.py
```
*** TODO - GPT 5 doesn't get the verified true/false flag right, should override with boolean checks ***


# Training, Saving, and Evaluating Model:

```bash
python src/train.py --batch-size 8 --epochs 1 --data-path ./data/gpt4o_labelled.json --output-dir ./models/distilbert-intent-classifier
```

Or transitional error weighted training:
```bash
python src/weighted_train.py --batch-size 8 --epochs 1 --data-path ./data/gpt4o_labelled.json --output-dir ./models/distilbert-intent-classifier
```

## Baby Example

```bash
============================================================
FINAL EVALUATION RESULTS
============================================================
VALIDATION SET:
  Accuracy: 0.2400
  F1 Macro: 0.0298
  F1 Weighted: 0.0929

TEST SET (HELD-OUT):
  Accuracy: 0.2533
  F1 Macro: 0.0337
  F1 Weighted: 0.1024

Test Set Classification Report (75 samples):
                                       precision    recall  f1-score   support

                        academic_help       0.25      1.00      0.40        19
    personal_writing_or_communication       0.00      0.00      0.00         0
                  writing_and_editing       0.00      0.00      0.00        12
                        write_fiction       0.00      0.00      0.00         0
                        how_to_advice       0.00      0.00      0.00         6
                    creative_ideation       0.00      0.00      0.00         5
                          translation       0.00      0.00      0.00         1
                 computer_programming       0.00      0.00      0.00         3
                 purchasable_products       0.00      0.00      0.00         0
                  cooking_and_recipes       0.00      0.00      0.00         0
   health_fitness_beauty_or_self_care       0.00      0.00      0.00         0
                        specific_info       0.00      0.00      0.00         9
               greetings_and_chitchat       0.00      0.00      0.00         3
relationships_and_personal_reflection       0.00      0.00      0.00         9
                  games_and_role_play       0.00      0.00      0.00         4
         media_generation_or_analysis       0.00      0.00      0.00         3
                              unclear       0.00      0.00      0.00         0
                                other       0.00      0.00      0.00         1

                             accuracy                           0.25        75
                            macro avg       0.01      0.06      0.02        75
                         weighted avg       0.06      0.25      0.10        75
```

**Precision**: Of all predictions for this class, what % were correct?
- `academic_help: 0.25` = 25% of "academic_help" predictions were actually academic_help
- High precision = low false positives

**Recall**: Of all actual instances of this class, what % did we catch?
- `academic_help: 1.00` = We caught 100% of all academic_help examples
- High recall = low false negatives

**F1-Score**: Harmonic mean of precision and recall (0-1, higher is better)
- Balances precision and recall
- `academic_help: 0.40` = Reasonable balance despite low precision

**Support**: Number of actual examples of this class in the test set
- `academic_help: 19` = 19 examples in the 75-sample test set

## Overall Metrics

**Accuracy**: Overall correct predictions / total predictions
- `0.2533` = 25.33% of predictions were correct

**Macro Average**: Simple average across all classes (treats all classes equally)
- `F1 Macro: 0.0337` = Very low, heavily penalized by classes with 0 performance

**Weighted Average**: Average weighted by class frequency (realistic for imbalanced data)
- `F1 Weighted: 0.1024` = More representative score for imbalanced datasets

## Interpreting These Results

### 🚨 Current Model Performance: **Poor**
- **25% accuracy** on 18-class classification (random guessing = 5.6%)
- **Model is severely underfitting** - only learned to predict one class (`academic_help`)
- **Classic "majority class bias"** - predicting most common class for everything

### 🔍 Root Causes
1. **Too little training data** (499 samples ÷ 18 classes = ~28 samples per class)
2. **Severe class imbalance** (some classes have 0 examples in test set)
3. **Insufficient training** (only 2 epochs, model hasn't converged)
4. **Complex task** (18 classes require more data to distinguish)

### 📈 Expected Improvements with Scale
- **Target dataset**: 10,000+ samples should dramatically improve performance
- **Expected performance**: 70-85% accuracy typical for well-trained intent classifiers
- **Balanced data**: Equal representation across classes will help significantly

### 💡 Quick Diagnostic Tips
- **All 0.00 metrics** = Model never predicted that class
- **High recall, low precision** = Model over-predicts this class  
- **Low recall, high precision** = Model is too conservative with this class
- **Macro vs Weighted avg gap** = Indicates class imbalance issues

Remember: These results are from a tiny development dataset. Performance should improve dramatically with your full 10k+ production dataset!


# Export & Quantize Model with ONNX:

```bash
python src/export_onnx.py --model-path models/distilbert-intent-classifier --output-path models/onnx-intent-classifier --benchmark --test
```


# Serving Model with FastAPI:

Local start:
```bash
python src/serve_fastapi.py --model-path models/onnx-intent-classifier --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

Test ping: 
```bash
curl -X POST "http://localhost:8000/classify/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "the most visited place in Wisconsin?"},
      {"role": "assistant", "content": "The most visited place in Wisconsin is the Wisconsin Dells..."},
      {"role": "user", "content": "where is Temecula?"},
      {"role": "assistant", "content": "Temecula is a city located in Southern California..."}
    ],
    "top_k": 3
  }'
```

Should Return: 
```bash
{"original_input":{"messages":[{"role":"user","content":"the most visited place in Wisconsin?"},{"role":"assistant","content":"The most visited place in Wisconsin is the Wisconsin Dells..."},{"role":"user","content":"where is Temecula?"},{"role":"assistant","content":"Temecula is a city located in Southern California..."}],"formatted_chat":null},"intent_class":"academic_help","confidence":0.14675700664520264,"predictions":[{"intent":"academic_help","confidence":0.14675700664520264},{"intent":"specific_info","confidence":0.08774775266647339},{"intent":"writing_and_editing","confidence":0.08627524971961975}],"processing_time_ms":67.00412509962916,"model_version":"unknown"}%
```


Some helpful Docker commands: 
```bash
# Make executable
chmod +x deployment/deploy.sh

# Deploy/redeploy
./deployment/deploy.sh

# Check status
cd deployment && docker compose ps

# View logs
cd deployment && docker compose logs -f

# Stop the API
cd deployment && docker compose down

# Rebuild after code changes
cd deployment && docker-compose build && docker-compose up -d
```
