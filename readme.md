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

Begin with a `.parquet` file in `data/`, 

```bash 
python src/convert_parquet_to_json.py
```
To create an appropriately structures `records.json` in `data/`. If the autorename fails, do it manually. Then,

```bash 
python src/extract_sequences.py
```
The first step in preprocessing. Then, 

```bash
python src/preprocess_sequences.py
```
You might need to change the `max_tokens` arg here to 512 or 1024 depending on your target token window. 

# Get gemini 2.5 Flash labels: 

```bash
python src/get_gemini_flash_labels.py
```

# Get GPT-4o-mini labels and soft labels: 
```bash 
python src/get_gpt_4o_mini_logprobs_unverified.py
```
To verify labels with gemini flash run (will save a NEW json of only AGREED on samples). 
```bash 
python src/compare_and_filter.py
```
Provides some reporting/visibility on agreement rates and class breakdowns in 'true' data. 

# Model creation and training

For 512 token window distilBERT and BERTs, run: 
```bash
python src/train_distilBERT.py
```
You can modify the base model you train by changing the `model_name` string in the config. 

