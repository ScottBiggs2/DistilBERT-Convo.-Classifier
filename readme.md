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

You will need to change the `max_tokens` arg here to 512 or 1024 depending on your target token window. Currently defaults to 1024.
```bash 
python src/preprocess_sequences.py
```
Very simple extraction script. 

# Initial Labelling Approach: GPT 5 then 4o:
However, now it is GPT 4o and GPT 4o-mini
```bash 
python src/get_gpt_4o_labels.py 
```
Or

```bash
python src/get_gemini_flash_labels.py
```

Defunct: 
```bash
python src/get_gpt_4o_mini_logprobs_verified.py
```

# Alternative Labbelling Approach: GPT 4o-mini only: 

```bash 
python src/get_gpt_4o_mini_logprobs_unverified.py
```
To verify labels with gemini flash run (will save a NEW json of only AGREED on samples). 
```bash 
python src/compare_and_filter.py
```
Provides some reporting/visibility on agreement rates and class breakdowns in 'true' data. 

# Then... 

For 512 token window distilBERT and BERTs, run: 
```bash
python src/train_distilBERT.py
```
You can modify the base model you train by changing the `model_name` string in the config. 


For distilBERT with 1024 tokens window,
```bash
python src/train_1024_distilBERT.py
```

For BERT with 1024 tokens window,
```bash
python src/train_1024_BERT.py
```


# To extract from Parquet: 
```bash 
python src/convert_parquet_to_json.py
```
Currently this pulls the top 20k rows. To change remove `.head(20000)` after parquet reading. Then run from the top.

# Some file conversions: 

Example 
```bash
python src/convert_jsonl_to_json.py data/gemini_2.5_flash_labelled.jsonl data/gemini_2.5_flash_labelled.json
```

Clean JSON files of errors: 
```bash
python src/filter_errors.py data/gemini_2.5_flash_labelled.json data/cleaned_gemini_2.5_flash_labelled.json
```

If you run them seperately: 
```bash
python src/compare_and_filter.py
```

# Save as ONNX, in case autosave fails:
```bash
python src/export_to_onnx.py --model-path models/distilbert_intent_classifier --output-path models/onnx_intent_classifier --benchmark --test
```