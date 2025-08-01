#!/usr/bin/env python3
"""
Simple Local Deployment Script for Hugging Face Hub

This script uploads your trained model and updates documentation on HF Hub.
Run this locally after training your model.

Usage:
    python deploy_to_hf.py

Requirements:
    - HF_TOKEN environment variable or ~/.cache/huggingface/token
    - Trained model in outputs/fixed_run/model_best.pth
"""

import os
import sys
from pathlib import Path
import json
import argparse
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from huggingface_hub import HfApi, create_repo
    from image_quality_fusion.models.huggingface_wrapper import ImageQualityFusionHF
    from image_quality_fusion.utils.paths import get_project_root
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you have installed the requirements:")
    print("   pip install huggingface_hub torch torchvision open-clip-torch opencv-python pillow")
    sys.exit(1)


class SimpleHFDeployer:
    """Simple local deployment to Hugging Face Hub."""
    
    def __init__(self, repo_name: str = "image-quality-fusion"):
        self.project_root = get_project_root()
        self.repo_name = repo_name
        self.model_path = self.project_root / "outputs" / "fixed_run" / "model_best.pth"
        self.api = HfApi()
        
        # Get username from git config or environment
        self.username = self._get_username()
        self.repo_id = f"{self.username}/{self.repo_name}"
        
        print(f"🎯 Target repository: {self.repo_id}")
        print(f"📁 Project root: {self.project_root}")
        print(f"🤖 Model path: {self.model_path}")
    
    def _get_username(self) -> str:
        """Get username from git config, environment, or prompt user."""
        # Try git config first
        try:
            import subprocess
            result = subprocess.run(['git', 'config', 'user.name'], 
                                   capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0 and result.stdout.strip():
                git_username = result.stdout.strip().lower().replace(' ', '')
                print(f"📝 Using git username: {git_username}")
                return git_username
        except:
            pass
        
        # Try environment variable
        if 'HF_USERNAME' in os.environ:
            return os.environ['HF_USERNAME']
        
        # Prompt user
        username = input("Enter your Hugging Face username: ").strip()
        if not username:
            print("❌ Username is required")
            sys.exit(1)
        return username
    
    def check_auth(self) -> bool:
        """Check if user is authenticated with Hugging Face."""
        try:
            user_info = self.api.whoami()
            print(f"✅ Authenticated as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("💡 Please login with: huggingface-cli login")
            return False
    
    def check_model_exists(self) -> bool:
        """Check if the trained model exists."""
        if not self.model_path.exists():
            print(f"❌ Model not found: {self.model_path}")
            print("💡 Train your model first with:")
            print("   python src/image_quality_fusion/training/train_fusion.py --help")
            return False
        
        print(f"✅ Model found: {self.model_path}")
        return True
    
    def create_repository(self) -> bool:
        """Create or ensure repository exists."""
        try:
            # Try to get repo info (checks if exists)
            self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            print(f"✅ Repository already exists: {self.repo_id}")
            return True
        except:
            # Repository doesn't exist, create it
            try:
                create_repo(
                    repo_id=self.repo_id,
                    repo_type="model",
                    private=False,
                    exist_ok=True
                )
                print(f"✅ Created repository: {self.repo_id}")
                return True
            except Exception as e:
                print(f"❌ Failed to create repository: {e}")
                return False
    
    def prepare_model(self) -> Optional[Path]:
        """Convert local model to HF format and save temporarily."""
        try:
            print("🔄 Converting model to HF format...")
            
            # Load the model using our wrapper
            hf_model = ImageQualityFusionHF.from_pretrained_fusion(str(self.model_path))
            
            # Save in HF format to temporary directory
            temp_dir = self.project_root / "temp_hf_model"
            temp_dir.mkdir(exist_ok=True)
            
            # Save model and config
            hf_model.save_pretrained(str(temp_dir))
            
            print(f"✅ Model prepared in HF format: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            print(f"❌ Failed to prepare model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_model_card(self) -> str:
        """Create model card content."""
        # Try to get model metadata
        metadata = {}
        try:
            import torch
            checkpoint = torch.load(self.model_path, map_location='cpu')
            metadata = checkpoint.get('metadata', {})
        except:
            pass
        
        metrics = metadata.get('metrics', {
            'correlation': 0.52,
            'r2_score': 0.25,
            'mae': 1.43
        })
        
        model_card = f"""---
license: mit
tags:
- image-quality
- computer-vision
- brisque
- aesthetic
- clip
- fusion
language:
- en
pipeline_tag: image-classification
library_name: pytorch
datasets:
- spaq
---

# Image Quality Fusion Model

A multi-modal image quality assessment system that combines BRISQUE, Aesthetic Predictor, and CLIP features to predict human-like quality judgments.

## Quick Start

```python
from huggingface_hub import PyTorchModelHubMixin

# Load model
model = PyTorchModelHubMixin.from_pretrained("{self.repo_id}")

# Predict quality for an image
quality_score = model.predict_quality("path/to/your/image.jpg")
print(f"Quality score: {{quality_score:.2f}}/10")

# Batch prediction
scores = model.predict_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
```

## Model Description

This model fuses three complementary approaches:

- **🔧 BRISQUE (OpenCV)**: Technical quality assessment
- **🎨 Aesthetic Predictor**: Visual appeal using LAION-trained CLIP features  
- **🧠 CLIP**: Semantic understanding (ViT-B-32)

## Performance

Trained on SPAQ dataset (11,125 smartphone images):

| Metric | Value |
|--------|-------|
| **Correlation with humans** | {metrics.get('correlation', 0.52):.3f} |
| **R² Score** | {metrics.get('r2_score', 0.25):.3f} |
| **Mean Absolute Error** | {metrics.get('mae', 1.43):.2f} points |

## Architecture

```
Input Image
    ├── OpenCV BRISQUE → Technical Quality Score
    ├── Aesthetic Predictor → Aesthetic Score 
    └── CLIP ViT-B-32 → Semantic Features (512-dim)
                ↓
        Deep Fusion Network
                ↓
        Human-like Quality Score (1-10)
```

## Usage Notes

- Optimized for smartphone photography
- Accepts PIL Images, numpy arrays, or file paths
- Returns quality scores on 1-10 scale
- Model size: ~100MB

## Citation

If you use this model, please cite:

```bibtex
@misc{{image-quality-fusion,
  title={{Image Quality Fusion: Multi-Modal Assessment}},
  author={{{self.username}}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{self.repo_id}}}}}
}}
```

## License

MIT License
"""
        return model_card
    
    def upload_model(self, model_dir: Path) -> bool:
        """Upload model and documentation to HF Hub."""
        try:
            print("🚀 Uploading to Hugging Face Hub...")
            
            # Create model card
            model_card = self.create_model_card()
            readme_path = model_dir / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            # Upload all files in the model directory
            self.api.upload_folder(
                folder_path=str(model_dir),
                repo_id=self.repo_id,
                repo_type="model",
                commit_message="Update model and documentation"
            )
            
            print(f"✅ Successfully uploaded to: https://huggingface.co/{self.repo_id}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self, temp_dir: Optional[Path]):
        """Clean up temporary files."""
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files")
    
    def deploy(self) -> bool:
        """Main deployment process."""
        print("🚀 Starting Hugging Face Deployment")
        print("=" * 50)
        
        # Pre-flight checks
        if not self.check_auth():
            return False
        
        if not self.check_model_exists():
            return False
        
        if not self.create_repository():
            return False
        
        # Prepare and upload model
        temp_dir = self.prepare_model()
        if not temp_dir:
            return False
        
        try:
            success = self.upload_model(temp_dir)
            return success
        finally:
            self.cleanup(temp_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy model to Hugging Face Hub")
    parser.add_argument("--repo-name", default="image-quality-fusion",
                       help="Name of the HF repository")
    parser.add_argument("--dry-run", action="store_true",
                       help="Check setup without uploading")
    
    args = parser.parse_args()
    
    deployer = SimpleHFDeployer(repo_name=args.repo_name)
    
    if args.dry_run:
        print("🔍 Dry run mode - checking setup...")
        success = (deployer.check_auth() and 
                  deployer.check_model_exists() and 
                  deployer.create_repository())
        if success:
            print("✅ Setup looks good! Run without --dry-run to deploy.")
        return success
    
    success = deployer.deploy()
    
    if success:
        print("\n🎉 Deployment Complete!")
        print(f"🌐 Your model: https://huggingface.co/{deployer.repo_id}")
        print("📝 Test it in the HF Hub interface")
    else:
        print("\n❌ Deployment failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)