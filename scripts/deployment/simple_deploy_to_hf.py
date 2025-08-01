#!/usr/bin/env python3
"""
Simple deployment script that doesn't require the HF wrapper
Just uploads raw model files to HF Hub
"""
import sys
from pathlib import Path
import json
import torch
from huggingface_hub import HfApi, upload_file, create_repo


def simple_deploy_to_hf(
    model_path: str,
    repo_name: str, 
    hf_username: str,
    commit_message: str = "Upload model"
):
    """
    Simple deployment without wrapper dependency
    """
    print("🚀 Simple deployment to Hugging Face Hub...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    api = HfApi()
    full_repo_name = f"{hf_username}/{repo_name}"
    
    try:
        # Create repo if it doesn't exist
        create_repo(full_repo_name, exist_ok=True)
        print(f"✅ Repository ready: {full_repo_name}")
    except Exception as e:
        print(f"Repository exists or created: {e}")
    
    # Load model to get metadata
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create simple config
    config = {
        "model_type": "image-quality-fusion",
        "task": "image-quality-assessment", 
        "framework": "pytorch",
        "license": "mit"
    }
    
    # Add model metadata if available
    if 'metadata' in checkpoint:
        config.update(checkpoint['metadata'].get('config', {}))
    
    # Upload config
    config_path = "/tmp/config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=full_repo_name,
        commit_message=f"{commit_message} - config"
    )
    print("✅ Config uploaded")
    
    # Upload model directly
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="pytorch_model.bin", 
        repo_id=full_repo_name,
        commit_message=f"{commit_message} - model"
    )
    print("✅ Model uploaded")
    
    # Create simple README
    readme_content = f"""---
license: mit
tags:
- image-quality
- pytorch
---

# {repo_name}

Image Quality Fusion model for predicting human-like quality scores.

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download("{full_repo_name}", "pytorch_model.bin")
config_path = hf_hub_download("{full_repo_name}", "config.json") 

# Load model
checkpoint = torch.load(model_path, map_location='cpu')
# Your custom loading code here...
```

Deployed automatically from GitHub Actions.
"""
    
    readme_path = "/tmp/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=full_repo_name, 
        commit_message=f"{commit_message} - readme"
    )
    print("✅ README uploaded")
    
    print(f"🎉 Simple deployment complete: https://huggingface.co/{full_repo_name}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--repo_name', required=True)
    parser.add_argument('--username', required=True)
    parser.add_argument('--commit_message', default='Upload model')
    
    args = parser.parse_args()
    
    simple_deploy_to_hf(
        args.model_path,
        args.repo_name,
        args.username, 
        args.commit_message
    )