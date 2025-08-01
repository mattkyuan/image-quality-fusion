#!/usr/bin/env python3
"""
SPAQ Dataset Preprocessing Script

Wrapper script for preprocessing SPAQ dataset with standard settings.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Preprocess SPAQ dataset with standard settings"""
    
    # Check if SPAQ dataset exists
    spaq_dir = Path("datasets/spaq")
    if not spaq_dir.exists():
        print("❌ SPAQ dataset not found at datasets/spaq/")
        print("Please download the dataset first:")
        print("  python scripts/download_spaq.py")
        return 1
    
    print("🔄 Preprocessing SPAQ dataset...")
    print(f"Input: {spaq_dir}")
    print("Output: datasets/processed/spaq/")
    print()
    
    # Run SPAQ preprocessing
    cmd = [
        sys.executable, "src/image_quality_fusion/data/spaq_preprocessing.py",
        str(spaq_dir),
        "--output_dir", "datasets/processed/spaq",
        "--create_splits",
        "--normalize_quality",
        "--target_range", "1.0", "10.0"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ SPAQ preprocessing completed successfully!")
        print("\n🚀 Next step: Train the model")
        print("  python scripts/train_spaq.py")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Preprocessing failed with error code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())