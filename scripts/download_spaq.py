#!/usr/bin/env python3
"""
SPAQ Dataset Download Helper

This script provides instructions and utilities for downloading the SPAQ dataset.
Since the dataset is large (~several GB), it must be downloaded manually.
"""

import sys
from pathlib import Path

def print_download_instructions():
    """Print instructions for downloading SPAQ dataset"""
    print("=" * 60)
    print("SPAQ DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("The SPAQ dataset contains 11,125 images and must be downloaded manually.")
    print()
    print("📥 DOWNLOAD SOURCES:")
    print("1. Google Drive: https://drive.google.com/drive/u/1/folders/1wZ6HOHi5h43oxTe2yLYkFxwHPgJ9MwvT")
    print("2. Baidu Yun: https://pan.baidu.com/s/18YzAtXb4cGdBGAsxuEVBOw (Code: b29m)")
    print()
    print("📁 DOWNLOAD STRUCTURE:")
    print("The download should contain:")
    print("  • images/ (11,125 image files)")
    print("  • annotations.csv (quality scores)")
    print("  • attributes.csv (image attributes)")
    print("  • scenes.csv (scene categories)")
    print("  • exif_data.csv (camera metadata)")
    print()
    print("💾 SAVE LOCATION:")
    print("Extract/save the dataset to: datasets/spaq/")
    print()
    print("Expected final structure:")
    print("datasets/spaq/")
    print("├── images/           # 11,125 images")
    print("├── annotations.csv   # Quality scores")
    print("├── attributes.csv    # Image attributes")
    print("├── scenes.csv        # Scene labels")
    print("└── exif_data.csv     # EXIF metadata")
    print()
    print("🚀 NEXT STEPS:")
    print("After downloading, run:")
    print("  python scripts/preprocess_spaq.py")
    print()

def check_spaq_download():
    """Check if SPAQ dataset has been downloaded"""
    spaq_dir = Path("datasets/spaq")
    images_dir = spaq_dir / "images"
    annotations_file = spaq_dir / "annotations.csv"
    
    if not spaq_dir.exists():
        print("❌ SPAQ dataset directory not found: datasets/spaq/")
        return False
    
    if not images_dir.exists():
        print("❌ SPAQ images directory not found: datasets/spaq/images/")
        return False
    
    if not annotations_file.exists():
        print("❌ SPAQ annotations file not found: datasets/spaq/annotations.csv")
        return False
    
    # Count images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_count = sum(1 for f in images_dir.rglob('*') 
                     if f.suffix.lower() in image_extensions)
    
    print(f"✅ SPAQ dataset found with {image_count:,} images")
    
    if image_count < 10000:
        print("⚠️  Warning: Expected ~11,125 images, but found fewer.")
        print("   Please check if download is complete.")
        return False
    
    print("🎉 SPAQ dataset appears to be complete!")
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Check if dataset is downloaded
        if check_spaq_download():
            print("\n🚀 Ready to preprocess SPAQ dataset!")
            print("Run: python scripts/preprocess_spaq.py")
        else:
            print("\n📥 Please download SPAQ dataset first.")
            print_download_instructions()
    else:
        # Show download instructions
        print_download_instructions()

if __name__ == "__main__":
    main()