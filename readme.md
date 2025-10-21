*** NOT CURRENT TREE ***
topic-classifier/
├─ data/
│  ├─ records.json              # raw JSON files [DONE]
│  ├─ raw_sequences.json        # JON with extracted message seuqnces [DONE]
│  ├─ cleaned_sequences.json    # Cleaned conversation sequences [DONE]
│  ├─ gpt4o_labelled.json       # full labeled dataset [DONE]
│  └─ verified_gpt5.json        # small sample for dev [DONE]
├─ models/                      # saved models - this is also a mess - [DONE]
|  └─   distilbert-intent-classifier/
|       └─ checkpoint-XX/       # HF model checkpoints [DONE]                     
├─ src/
│  ├─ extract_sequences.py      # truncation, text extraction, heuristics [DONE]
│  ├─ preprocess_sequences.py   # batch preprocessing, tokenization, truncation, etc [DONE]
│  ├─ label_and_audit.py        # Label conversation data [DONE]
│  ├─ train.py                  # HF fine-tune script (train/eval)
│  ├─ export_onnx.py            # export & quantize
│  ├─ serve_fastapi.py          # ONNXRuntime FastAPI server
│  ├─ utils.py                  # helpers (metrics, dataset class)
│  └─ rules.py                  # small rule-based heuristics for hybrids
├─ requirements.txt
├─ README.md
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


# Training and Saving Model:

```bash
python src/train.py --batch-size 16 --epochs 3 --data-path ./data/gpt4o_labelled.json --output-dir ./models/distilbert-intent-classifier
```

# Model Eval on Test Set:

```bash
blah blah blah 
```

# Serving Model with FastAPI:

```bash
blah blah blah
```