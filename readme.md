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

```bash 
python src/gpt_5_label.py 
```

```bash
python src/get_gpt_4o_logprobs.py
```

```bash
python src/train_distilBERT.py
```