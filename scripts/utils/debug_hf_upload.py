#!/usr/bin/env python3
"""
Debug HF upload and try alternative upload method
"""
import sys
from pathlib import Path
import torch

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

from image_quality_fusion.models.huggingface_wrapper import ImageQualityFusionHF
from huggingface_hub import HfApi, upload_file


def debug_current_upload():
    """Debug the current upload status"""
    print("🔍 Debugging current HF upload...")
    
    try:
        api = HfApi()
        repo_info = api.repo_info("matthewyuan/image-quality-fusion")
        
        print(f"📊 Repository Info:")
        print(f"   Repo ID: {repo_info.id}")
        print(f"   Created: {repo_info.created_at}")
        print(f"   Last Modified: {repo_info.last_modified}")
        
        # List files in repo
        files = api.list_repo_files("matthewyuan/image-quality-fusion")
        print(f"\n📁 Files in repository:")
        for file in files:
            print(f"   - {file}")
            
        return repo_info
        
    except Exception as e:
        print(f"❌ Error getting repo info: {e}")
        return None


def try_direct_upload():
    """Try uploading the model file directly"""
    print("\n🚀 Trying direct file upload...")
    
    model_path = "outputs/fixed_run/model_best.pth"
    
    try:
        # Upload the raw model file directly
        api = HfApi()
        
        print("📤 Uploading pytorch_model.bin directly...")
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="pytorch_model.bin",
            repo_id="matthewyuan/image-quality-fusion",
            commit_message="Add model weights directly"
        )
        
        print("✅ Direct upload completed!")
        return True
        
    except Exception as e:
        print(f"❌ Direct upload failed: {e}")
        return False


def try_safetensors_format():
    """Try converting to safetensors format (HF preferred)"""
    print("\n🔄 Trying safetensors format...")
    
    try:
        # Load the model
        model_path = "outputs/fixed_run/model_best.pth"
        checkpoint = torch.load(model_path, map_location='cpu')
        model_state = checkpoint['model_state_dict']
        
        # Create HF model wrapper
        hf_model = ImageQualityFusionHF.from_pretrained_fusion(model_path)
        
        # Try saving with safetensors
        temp_dir = Path("temp_hf_model")
        temp_dir.mkdir(exist_ok=True)
        
        print("💾 Saving with safetensors format...")
        hf_model.save_pretrained(temp_dir, safe_serialization=True)
        
        # Push the safetensors version
        print("📤 Pushing safetensors version...")
        hf_model.push_to_hub(
            "matthewyuan/image-quality-fusion", 
            commit_message="Update with safetensors format",
            safe_serialization=True
        )
        
        print("✅ Safetensors upload completed!")
        return True
        
    except Exception as e:
        print(f"❌ Safetensors upload failed: {e}")
        return False


def main():
    """Main debug function"""
    print("🛠️  Debugging Hugging Face Upload Issues")
    print("=" * 50)
    
    # Step 1: Debug current status
    repo_info = debug_current_upload()
    
    if not repo_info:
        print("❌ Cannot access repository - this might be the issue")
        return
    
    # Step 2: Try direct upload
    direct_success = try_direct_upload()
    
    if direct_success:
        print("\n✅ Direct upload worked! Checking status...")
        import time
        time.sleep(5)  # Wait a bit
        
        # Check again
        from scripts.monitor_hf_deployment import check_model_status
        check_model_status()
    else:
        # Step 3: Try safetensors format
        print("\n🔄 Direct upload failed, trying safetensors...")
        safetensors_success = try_safetensors_format()
        
        if safetensors_success:
            print("✅ Safetensors format uploaded!")
        else:
            print("❌ All upload methods failed")
            print("💡 This might be a temporary HF Hub issue")
            print("💡 Try again in 10-15 minutes or contact HF support")


if __name__ == "__main__":
    main()