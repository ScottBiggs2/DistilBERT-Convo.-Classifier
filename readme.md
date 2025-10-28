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


# Initial Labelling Approach: GPT 5 then 4o:
However, now it is GPT 4o and GPT 4o-mini
```bash 
python src/get_gpt_4o_labels.py 
```
Or

```bash
python src/get_gemini_flash_labels.py
```


```bash
python src/get_gpt_4o_mini_logprobs_verified.py
```

# Alternative Labbelling Approach: GPT 4o-mini only: 

```bash 
python src/get_gpt_4o_mini_logprobs_unverified.py
```

# Then... 

```bash
python src/train_distilBERT.py
```

# To extract from Parquet: 
```bash 
python src/convert_parquet_to_json.py
```
Then run from the top.

# JSONL to JSON: 

Example 
```bash
python src/convert_jsonl_to_json.py data/gemini_2.5_flash_labelled.jsonl data/gemini_2.5_flash_labelled.json
```

Clean JSON files of errors: 
```bash
python src/filter_errors.py data/gemini_2.5_flash_labelled.json data/cleaned_gemini_2.5_flash_labelled.json
```