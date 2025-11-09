"""
Upload trained models to HuggingFace Hub
Uploads model.safetensors and config files for distilled BERT models
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
ORG_NAME = os.getenv("HF_ORG_NAME")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")
if not ORG_NAME:
    raise ValueError("HF_ORG_NAME not found in .env file")

# Model configurations
MODELS = [
    {
        "local_path": "models/distilbert_distilled_BERT_0.0_50_epochs",
        "repo_name": "thrad-bert-conversation-classifier",
        "description": "BERT model for conversation intent classification",
    },
    {
        "local_path": "models/distilbert_distilled_alpha_0.0",
        "repo_name": "thrad-distilbert-conversation-classifier",
        "description": "DistilBERT model for conversation classification with hard labels",
    },
]


def upload_model(api: HfApi, model_config: dict, org_name: str):
    """Upload a single model to HuggingFace Hub"""
    
    local_path = Path(model_config["local_path"])
    repo_id = f"{org_name}/{model_config['repo_name']}"
    
    print(f"\n{'='*60}")
    print(f"Uploading: {repo_id}")
    print(f"From: {local_path}")
    print(f"{'='*60}\n")
    
    # Check if required files exist
    safetensors_path = local_path / "model.safetensors"
    config_path = local_path / "config.json"
    
    if not safetensors_path.exists():
        print(f"❌ Error: model.safetensors not found in {local_path}")
        return False
    
    if not config_path.exists():
        print(f"⚠️  Warning: config.json not found in {local_path}")
    
    try:
        # Create repository (will skip if exists)
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=HF_TOKEN,
        )
        print(f"✓ Repository created/verified: {repo_id}")
        
        # Upload model.safetensors
        api.upload_file(
            path_or_fileobj=str(safetensors_path),
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"✓ Uploaded model.safetensors")
        
        # Upload config.json if it exists
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=repo_id,
                repo_type="model",
                token=HF_TOKEN,
            )
            print(f"✓ Uploaded config.json")
        
        # Create a basic README
        readme_content = f"""---
license: apache-2.0
tags:
- text-classification
- distilbert
- conversation-classification
- knowledge-distillation
---

# {model_config['repo_name']}

{model_config['description']}

## Model Details

- **Base Architecture**: DistilBERT
- **Task**: Multi-class conversation intent classification

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Example inference
text = "Your conversation text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = outputs.logits.softmax(dim=-1)
```

"""
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"✓ Created README.md")
        
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{repo_id}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {str(e)}")
        return False


def main():
    """Main upload function"""
    
    # Initialize HF API
    api = HfApi()
    
    print(f"\n{'='*60}")
    print(f"HuggingFace Model Upload Script")
    print(f"Organization: {ORG_NAME}")
    print(f"{'='*60}")
    
    # Upload each model
    results = []
    for model_config in MODELS:
        success = upload_model(api, model_config, ORG_NAME)
        results.append((model_config["repo_name"], success))
    
    # Summary
    print(f"\n{'='*60}")
    print("Upload Summary")
    print(f"{'='*60}")
    for repo_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {repo_name}")
    print()


if __name__ == "__main__":
    main()