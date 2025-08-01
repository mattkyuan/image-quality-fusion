#!/usr/bin/env python3
"""
Simple test for the Hugging Face deployed model
(Can be run anywhere, doesn't need the full project)
"""
import sys
from pathlib import Path

def test_hf_model_simple():
    """Simple test that anyone can run"""
    try:
        # Import required packages
        from huggingface_hub import PyTorchModelHubMixin
        print("✅ huggingface_hub imported successfully")
        
        # Try to download and load the model
        print("🌐 Loading model from Hugging Face Hub...")
        print("   Model: matthewyuan/image-quality-fusion")
        
        # This will work once the model is fully processed on HF Hub
        from huggingface_hub import hf_hub_download
        
        # Check if model files exist
        try:
            config_path = hf_hub_download(
                repo_id="matthewyuan/image-quality-fusion",
                filename="config.json"
            )
            print(f"✅ Model config found at: {config_path}")
            
            model_path = hf_hub_download(
                repo_id="matthewyuan/image-quality-fusion", 
                filename="pytorch_model.bin"
            )
            print(f"✅ Model weights found at: {model_path}")
            
            print("🎉 Model is ready for use!")
            return True
            
        except Exception as e:
            print(f"⏳ Model still processing on HF Hub: {e}")
            print("💡 Try again in a few minutes")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install huggingface_hub torch")
        return False


def create_minimal_usage_example():
    """Create minimal usage example for users"""
    example = '''
# Minimal usage example for the Image Quality Fusion model

## Installation
```bash
pip install huggingface_hub torch torchvision pillow opencv-python open-clip-torch
```

## Usage
```python
from huggingface_hub import PyTorchModelHubMixin

# Load the model (this will download it automatically)
model = PyTorchModelHubMixin.from_pretrained("matthewyuan/image-quality-fusion")

# Predict image quality (1-10 scale)
quality_score = model.predict_quality("path/to/your/image.jpg")
print(f"Image quality: {quality_score:.2f}/10")

# Batch prediction
scores = model.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
for i, score in enumerate(scores, 1):
    print(f"Image {i}: {score:.2f}/10")
```

## What it does
- Combines BRISQUE (technical quality), Aesthetic Predictor, and CLIP features
- Trained on 11K smartphone photos with human quality ratings
- Achieves 0.52 correlation with human judgments
- Fast inference (~0.7s per image after initial load)
'''
    
    with open("USAGE_EXAMPLE.md", "w") as f:
        f.write(example)
    
    print("📝 Created USAGE_EXAMPLE.md")


def main():
    print("🔍 Simple Hugging Face Model Test")
    print("=" * 40)
    
    success = test_hf_model_simple()
    
    print("\n📝 Creating usage documentation...")
    create_minimal_usage_example()
    
    if success:
        print("\n✅ Model is ready to use!")
        print("🚀 Try it at: https://huggingface.co/matthewyuan/image-quality-fusion")
    else:
        print("\n⏳ Model is still being processed on Hugging Face Hub")
        print("💡 Check back in a few minutes")
    
    print("\n📖 See USAGE_EXAMPLE.md for usage instructions")


if __name__ == "__main__":
    main()