#!/usr/bin/env python3
"""
Monitor Hugging Face model deployment status
"""
import time
import sys
from datetime import datetime

def check_model_status():
    """Check if model is ready on HF Hub"""
    try:
        from huggingface_hub import hf_hub_download
        
        # Try to download key files
        files_to_check = [
            "config.json",
            "pytorch_model.bin",
            "model_info.json"
        ]
        
        ready_files = []
        for filename in files_to_check:
            try:
                path = hf_hub_download(
                    repo_id="matthewyuan/image-quality-fusion",
                    filename=filename
                )
                ready_files.append(filename)
                print(f"✅ {filename} - Ready")
            except Exception as e:
                print(f"⏳ {filename} - Not ready ({type(e).__name__})")
        
        progress = len(ready_files) / len(files_to_check) * 100
        print(f"📊 Progress: {progress:.0f}% ({len(ready_files)}/{len(files_to_check)} files)")
        
        if len(ready_files) == len(files_to_check):
            print("🎉 Model is fully ready!")
            return True
        else:
            print(f"⏳ Still processing... {len(files_to_check) - len(ready_files)} files remaining")
            return False
            
    except ImportError:
        print("❌ huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False


def test_model_functionality():
    """Test if model can be loaded and used"""
    try:
        from huggingface_hub import PyTorchModelHubMixin
        from src.image_quality_fusion.models.huggingface_wrapper import ImageQualityFusionHF
        
        print("🧪 Testing model functionality...")
        model = ImageQualityFusionHF.from_pretrained("matthewyuan/image-quality-fusion")
        print("✅ Model loads successfully!")
        
        # Test with a dummy prediction if possible
        info = model.get_model_info()
        print(f"📊 Model: {info['name']} v{info['version']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model functionality test failed: {e}")
        return False


def monitor_deployment(check_interval=60, max_checks=30):
    """
    Monitor deployment with periodic checks
    
    Args:
        check_interval: Seconds between checks (default: 60)
        max_checks: Maximum number of checks before giving up (default: 30)
    """
    print("🔍 Monitoring Hugging Face model deployment...")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Checking every {check_interval} seconds")
    print("🛑 Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    for check_num in range(1, max_checks + 1):
        print(f"\n📋 Check #{check_num}/{max_checks} at {datetime.now().strftime('%H:%M:%S')}")
        
        # Check file availability
        files_ready = check_model_status()
        
        if files_ready:
            # Test functionality
            functional = test_model_functionality()
            
            if functional:
                print("\n🎉🎉 MODEL IS FULLY READY! 🎉🎉")
                print("🚀 You can now use: ImageQualityFusionHF.from_pretrained('matthewyuan/image-quality-fusion')")
                print("🌐 Visit: https://huggingface.co/matthewyuan/image-quality-fusion")
                return True
            else:
                print("⚠️  Files ready but model not functional yet...")
        
        if check_num < max_checks:
            print(f"⏳ Waiting {check_interval}s for next check...")
            try:
                time.sleep(check_interval)
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                return False
    
    print(f"\n⏰ Reached maximum checks ({max_checks})")
    print("💡 Model might need more time. Try checking manually later.")
    return False


def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor HF model deployment')
    parser.add_argument('--interval', '-i', type=int, default=60, 
                       help='Check interval in seconds (default: 60)')
    parser.add_argument('--max-checks', '-m', type=int, default=30,
                       help='Maximum number of checks (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Check once and exit')
    
    args = parser.parse_args()
    
    if args.once:
        print("🔍 Single status check...")
        files_ready = check_model_status()
        if files_ready:
            test_model_functionality()
    else:
        monitor_deployment(args.interval, args.max_checks)


if __name__ == "__main__":
    main()