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