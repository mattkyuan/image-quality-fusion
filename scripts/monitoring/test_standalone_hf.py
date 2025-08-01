#!/usr/bin/env python3
"""
Standalone test of HF model (without project dependencies)
"""

def test_standalone_hf_model():
    """Test the HF model as an external user would"""
    try:
        print("🌐 Testing standalone HF model loading...")
        
        # This is how external users would load it
        from huggingface_hub import hf_hub_download
        import torch
        import json
        
        # Download model files
        config_path = hf_hub_download(
            repo_id="matthewyuan/image-quality-fusion",
            filename="config.json"
        )
        
        model_path = hf_hub_download(
            repo_id="matthewyuan/image-quality-fusion", 
            filename="pytorch_model.bin"
        )
        
        print("✅ Model files downloaded successfully")
        print(f"   Config: {config_path}")
        print(f"   Model: {model_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Config loaded: {config}")
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model weights loaded: {list(checkpoint.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        return False


def check_hf_hub_integration():
    """Check if the model integrates properly with HF Hub"""
    try:
        from huggingface_hub import PyTorchModelHubMixin
        
        print("🔗 Testing HF Hub integration...")
        
        # This would be the ideal usage, but requires the wrapper class
        # model = PyTorchModelHubMixin.from_pretrained("matthewyuan/image-quality-fusion")
        
        # For now, just check if files are accessible
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Check model card
        try:
            readme = api.hf_hub_download(
                repo_id="matthewyuan/image-quality-fusion",
                filename="README.md"
            )
            print("✅ README.md accessible")
        except:
            print("⚠️  README.md not found")
        
        # List all files
        files = api.list_repo_files("matthewyuan/image-quality-fusion")
        print(f"✅ Repository contains {len(files)} files:")
        for f in files:
            print(f"   - {f}")
            
        return True
        
    except Exception as e:
        print(f"❌ HF Hub integration test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🔍 Checking HF Model Status")
    print("=" * 40)
    
    # Test 1: Standalone file access
    standalone_ok = test_standalone_hf_model()
    print()
    
    # Test 2: HF Hub integration
    integration_ok = check_hf_hub_integration()
    print()
    
    # Summary
    if standalone_ok and integration_ok:
        print("🎉 Model is fully deployed and accessible!")
        print("📈 Status: READY FOR USE")
        print("🌐 URL: https://huggingface.co/matthewyuan/image-quality-fusion")
    elif standalone_ok:
        print("⚠️  Model files are ready but integration needs work")
        print("📈 Status: PARTIALLY READY")
    else:
        print("❌ Model has issues")
        print("📈 Status: NOT READY")


if __name__ == "__main__":
    main()