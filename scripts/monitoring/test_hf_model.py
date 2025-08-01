#!/usr/bin/env python3
"""
Test script for the deployed Hugging Face model
"""
import sys
from pathlib import Path
import time

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from huggingface_hub import PyTorchModelHubMixin
    from image_quality_fusion.models.huggingface_wrapper import ImageQualityFusionHF
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure huggingface_hub is installed: pip install huggingface_hub")
    sys.exit(1)


def test_local_model():
    """Test the local model wrapper"""
    print("🧪 Testing local model wrapper...")
    
    try:
        # Load from local checkpoint
        model_path = "outputs/fixed_run/model_best.pth"
        if not Path(model_path).exists():
            print(f"❌ Local model not found at: {model_path}")
            return False
        
        model = ImageQualityFusionHF.from_pretrained_fusion(model_path)
        print("✅ Local model loaded successfully")
        
        # Get model info
        info = model.get_model_info()
        print(f"📊 Model: {info['name']} v{info['version']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local model test failed: {e}")
        return False


def test_huggingface_model():
    """Test the deployed Hugging Face model"""
    print("🌐 Testing Hugging Face model...")
    
    try:
        # Load from Hugging Face Hub
        model = ImageQualityFusionHF.from_pretrained("matthewyuan/image-quality-fusion")
        print("✅ HF model loaded successfully")
        
        return model
        
    except Exception as e:
        print(f"❌ HF model test failed: {e}")
        print("💡 The model might still be processing on HF Hub")
        return None


def test_with_sample_images(model):
    """Test model with sample images"""
    print("📸 Testing with sample images...")
    
    # Look for demo images
    demo_paths = [
        "datasets/demo",
        "datasets/spaq/images",
        "data/images"
    ]
    
    sample_images = []
    for demo_path in demo_paths:
        demo_dir = Path(demo_path)
        if demo_dir.exists():
            # Get first few images
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in image_extensions:
                sample_images.extend(list(demo_dir.glob(ext))[:3])
                if len(sample_images) >= 3:
                    break
            if sample_images:
                break
    
    if not sample_images:
        print("⚠️  No sample images found in demo directories")
        print("💡 Add some test images to datasets/demo/ to test predictions")
        return
    
    print(f"Found {len(sample_images)} sample images")
    
    # Test predictions
    for i, image_path in enumerate(sample_images[:3], 1):
        try:
            print(f"\n🖼️  Testing image {i}: {image_path.name}")
            
            start_time = time.time()
            score = model.predict_quality(str(image_path))
            inference_time = time.time() - start_time
            
            print(f"   Quality Score: {score:.2f}/10")
            print(f"   Inference Time: {inference_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed to process {image_path.name}: {e}")


def test_batch_prediction(model):
    """Test batch prediction"""
    print("\n📦 Testing batch prediction...")
    
    # Create dummy image paths for testing
    demo_dir = Path("datasets/demo")
    if demo_dir.exists():
        image_paths = list(demo_dir.glob("*.jpg"))[:3]
        if image_paths:
            try:
                start_time = time.time()
                scores = model.predict_batch([str(p) for p in image_paths])
                batch_time = time.time() - start_time
                
                print(f"✅ Batch prediction successful")
                print(f"   Images processed: {len(scores)}")
                print(f"   Total time: {batch_time:.3f}s")
                print(f"   Avg time per image: {batch_time/len(scores):.3f}s")
                
                for path, score in zip(image_paths, scores):
                    print(f"   {path.name}: {score:.2f}/10")
                    
            except Exception as e:
                print(f"❌ Batch prediction failed: {e}")
        else:
            print("⚠️  No images found for batch testing")
    else:
        print("⚠️  Demo directory not found for batch testing")


def create_usage_example():
    """Create a usage example script"""
    example_code = '''#!/usr/bin/env python3
"""
Example usage of the Image Quality Fusion model from Hugging Face Hub
"""
from huggingface_hub import PyTorchModelHubMixin
from src.image_quality_fusion.models.huggingface_wrapper import ImageQualityFusionHF

def main():
    # Load model from Hugging Face Hub
    print("Loading model from Hugging Face Hub...")
    model = ImageQualityFusionHF.from_pretrained("matthewyuan/image-quality-fusion")
    
    # Example 1: Single image prediction
    image_path = "path/to/your/image.jpg"
    score = model.predict_quality(image_path)
    print(f"Quality score: {score:.2f}/10")
    
    # Example 2: Batch prediction
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    scores = model.predict_batch(image_paths)
    for path, score in zip(image_paths, scores):
        print(f"{path}: {score:.2f}/10")
    
    # Example 3: Get model information
    info = model.get_model_info()
    print(f"Model: {info['name']} v{info['version']}")
    print(f"Description: {info['description']}")

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/usage_example.py", "w") as f:
        f.write(example_code)
    print("📝 Created usage_example.py script")


def main():
    """Main test function"""
    print("🚀 Testing Image Quality Fusion Hugging Face Deployment")
    print("=" * 60)
    
    # Test 1: Local model
    local_success = test_local_model()
    print()
    
    # Test 2: Hugging Face model
    hf_model = test_huggingface_model()
    print()
    
    # Use whichever model loaded successfully
    test_model = hf_model if hf_model else None
    if not test_model and local_success:
        print("🔄 Falling back to local model for testing...")
        model_path = "outputs/fixed_run/model_best.pth"
        test_model = ImageQualityFusionHF.from_pretrained_fusion(model_path)
    
    if test_model:
        # Test 3: Sample image predictions
        test_with_sample_images(test_model)
        
        # Test 4: Batch predictions
        test_batch_prediction(test_model)
        
        print("\n✅ All tests completed!")
    else:
        print("❌ No model available for testing")
    
    # Create usage example
    print("\n📝 Creating usage example...")
    create_usage_example()
    
    print("\n🎉 Testing complete!")
    print("💡 Try the deployed model at: https://huggingface.co/matthewyuan/image-quality-fusion")


if __name__ == "__main__":
    main()