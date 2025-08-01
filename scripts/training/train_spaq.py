#!/usr/bin/env python3
"""
SPAQ Model Training Script

Wrapper script for training the fusion model on SPAQ dataset with optimized settings.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    """Train fusion model on SPAQ dataset"""
    
    # Check if preprocessed SPAQ data exists
    annotations_file = Path("datasets/processed/spaq/annotations_fusion.csv")
    if not annotations_file.exists():
        print("❌ Preprocessed SPAQ data not found!")
        print("Please preprocess the dataset first:")
        print("  python scripts/preprocess_spaq.py")
        return 1
    
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"spaq_baseline_{timestamp}"
    
    print("🚀 Starting SPAQ model training...")
    print(f"Experiment: {experiment_name}")
    print("Expected duration: 2-4 hours (with GPU), 8-12 hours (CPU)")
    print()
    
    # Training command with SPAQ-optimized parameters
    cmd = [
        sys.executable, "src/image_quality_fusion/training/train_fusion.py",
        "--image_dir", "datasets/spaq/images",
        "--annotations", "datasets/processed/spaq/annotations_fusion.csv",
        "--prepare_data",
        "--features_dir", "datasets/processed/spaq/features",
        
        # Model configuration
        "--model_type", "deep",
        "--hidden_dim", "512",
        "--dropout_rate", "0.4",
        "--output_range", "1.0", "10.0",
        
        # Training configuration
        "--epochs", "150",
        "--batch_size", "64",
        "--learning_rate", "0.0005",
        "--weight_decay", "0.001",
        "--patience", "25",
        
        # Output configuration
        "--output_dir", "experiments",
        "--experiment_name", experiment_name,
        
        # System configuration
        "--device", "auto",
        "--num_workers", "4"
    ]
    
    try:
        print("Command:", " ".join(cmd))
        print("\n" + "="*60)
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("✅ SPAQ training completed successfully!")
        print(f"\n📁 Results saved to: experiments/{experiment_name}/")
        print(f"📊 Model file: experiments/{experiment_name}/model_best.pth")
        print(f"📈 Training plots: experiments/{experiment_name}/training_plots.png")
        
        print("\n🚀 Next steps:")
        print("1. Check training results:")
        print(f"   ls experiments/{experiment_name}/")
        print("2. Test your model:")
        print(f"   python scripts/test_model.py experiments/{experiment_name}/model_best.pth")
        print("3. Move to production:")
        print(f"   cp experiments/{experiment_name}/model_best.pth models/production/fusion_spaq_v1.0.pth")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        print("\nTroubleshooting:")
        print("- Check GPU memory if using GPU")
        print("- Reduce batch_size if out of memory") 
        print("- Check dataset integrity")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())