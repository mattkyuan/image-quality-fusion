#!/usr/bin/env python3
"""
Deploy Image Quality Fusion model to Hugging Face Hub
"""
import os
import sys
from pathlib import Path

# Add project root to path and ensure imports work
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from image_quality_fusion.models.huggingface_wrapper import ImageQualityFusionHF


def deploy_model_to_hf(
    model_path: str,
    repo_name: str,
    hf_username: str,
    private: bool = False,
    commit_message: str = "Upload Image Quality Fusion model"
):
    """
    Deploy the trained model to Hugging Face Hub.
    
    Args:
        model_path: Path to trained model (.pth file)
        repo_name: Repository name on HF Hub
        hf_username: Your Hugging Face username
        private: Whether to make repo private
        commit_message: Commit message for upload
    """
    
    print("🚀 Deploying Image Quality Fusion model to Hugging Face Hub...")
    
    # Check if model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Check if already logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Logged in to Hugging Face as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Not logged in to Hugging Face: {e}")
        print("💡 Run: hf auth login")
        return
    
    # Load and wrap the model
    print("📦 Loading and wrapping model...")
    hf_model = ImageQualityFusionHF.from_pretrained_fusion(model_path)
    
    # Display model info
    info = hf_model.get_model_info()
    print(f"📊 Model Info:")
    print(f"   - Name: {info['name']}")
    print(f"   - Version: {info['version']}")
    perf = info.get('performance', {})
    print(f"   - Correlation: {perf.get('correlation', 'N/A')}")
    print(f"   - R² Score: {perf.get('r2_score', 'N/A')}")
    print(f"   - MAE: {perf.get('mae', 'N/A')}")
    
    # Create full repo name
    full_repo_name = f"{hf_username}/{repo_name}"
    
    # Upload to Hub
    print(f"☁️  Uploading to {full_repo_name}...")
    try:
        hf_model.push_to_hub(
            full_repo_name,
            private=private,
            commit_message=commit_message
        )
        print(f"🎉 Successfully deployed to: https://huggingface.co/{full_repo_name}")
        
        # Print usage example
        print(f"\n📖 Usage Example:")
        print(f"```python")
        print(f"from huggingface_hub import from_pretrained_keras")
        print(f"from PIL import Image")
        print(f"")
        print(f"# Load model")
        print(f"model = ImageQualityFusionHF.from_pretrained('{full_repo_name}')")
        print(f"")
        print(f"# Predict quality")
        print(f"score = model.predict_quality('path/to/image.jpg')")
        print(f"print(f'Quality score: {{score:.2f}}/10')")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        return False
    
    return True


def main():
    """Main deployment script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Image Quality Fusion model to Hugging Face Hub')
    parser.add_argument('--model_path', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--repo_name', required=True, help='Repository name on HF Hub')
    parser.add_argument('--username', required=True, help='Your Hugging Face username')
    parser.add_argument('--private', action='store_true', help='Make repository private')
    parser.add_argument('--commit_message', default='Upload Image Quality Fusion model', 
                       help='Commit message for upload')
    
    args = parser.parse_args()
    
    # Deploy model
    success = deploy_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_username=args.username,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if success:
        print("\n✅ Deployment completed successfully!")
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    # For direct execution with your model
    model_path = "outputs/fixed_run/model_best.pth"
    
    if len(sys.argv) == 1:
        print("🔧 Example Usage:")
        print(f"python {sys.argv[0]} --model_path {model_path} --repo_name image-quality-fusion --username YOUR_USERNAME")
        print("\n💡 Or run interactively:")
        
        # Interactive mode
        username = input("Enter your HF username: ").strip()
        repo_name = input("Enter repository name [image-quality-fusion]: ").strip() or "image-quality-fusion"
        private = input("Make repository private? [y/N]: ").strip().lower() == 'y'
        
        if username:
            deploy_model_to_hf(
                model_path=model_path,
                repo_name=repo_name,
                hf_username=username,
                private=private
            )
    else:
        main()