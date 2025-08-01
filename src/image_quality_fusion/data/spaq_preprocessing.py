#!/usr/bin/env python3
"""
SPAQ Dataset Preprocessing Script

This script converts the SPAQ (Smartphone Photography Attribute and Quality) dataset
into the format required by the Image Quality Fusion training pipeline.

SPAQ Dataset:
- 11,125 images from 66 smartphones
- Human quality scores (MOS - Mean Opinion Score)
- Image attributes: brightness, colorfulness, contrast, noisiness, sharpness
- Scene category labels
- EXIF metadata

Expected SPAQ directory structure:
spaq_dataset/
├── images/                # All image files
├── annotations.csv        # Main annotations file
├── attributes.csv         # Image attributes (optional)
├── scenes.csv            # Scene category labels (optional)
└── exif_data.csv         # EXIF metadata (optional)

Output format for fusion training:
annotations_fusion.csv with columns: image_path,human_score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Optional, Tuple, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPAQPreprocessor:
    """
    Preprocessor for SPAQ dataset to convert to fusion training format
    """
    
    def __init__(self, spaq_dir: str, output_dir: str = "./processed_spaq"):
        """
        Initialize SPAQ preprocessor
        
        Args:
            spaq_dir: Path to SPAQ dataset directory
            output_dir: Path to output processed data
        """
        self.spaq_dir = Path(spaq_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected SPAQ files
        self.images_dir = self.spaq_dir / "images"
        self.annotations_file = None  # Will be detected automatically
        self.attributes_file = None
        self.scenes_file = None
        self.exif_file = None
        
        logger.info(f"SPAQ directory: {self.spaq_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def discover_files(self) -> bool:
        """
        Automatically discover SPAQ annotation files
        
        Returns:
            bool: True if required files found
        """
        logger.info("🔍 Discovering SPAQ dataset files...")
        
        if not self.images_dir.exists():
            logger.error(f"Images directory not found: {self.images_dir}")
            return False
        
        # Common annotation file names in SPAQ dataset
        annotation_candidates = [
            "annotations.csv",
            "quality_scores.csv", 
            "mos_scores.csv",
            "labels.csv",
            "spaq_annotations.csv"
        ]
        
        for candidate in annotation_candidates:
            candidate_path = self.spaq_dir / candidate
            if candidate_path.exists():
                self.annotations_file = candidate_path
                logger.info(f"✅ Found annotations: {candidate}")
                break
        
        if not self.annotations_file:
            logger.warning("❌ Main annotations file not found. Please specify manually.")
            logger.info("Expected files: " + ", ".join(annotation_candidates))
            return False
        
        # Look for optional files
        optional_files = {
            "attributes": ["attributes.csv", "image_attributes.csv"],
            "scenes": ["scenes.csv", "scene_labels.csv", "categories.csv"],
            "exif": ["exif.csv", "exif_data.csv", "metadata.csv"]
        }
        
        for file_type, candidates in optional_files.items():
            for candidate in candidates:
                candidate_path = self.spaq_dir / candidate
                if candidate_path.exists():
                    if file_type == "attributes":
                        self.attributes_file = candidate_path
                    elif file_type == "scenes":
                        self.scenes_file = candidate_path
                    elif file_type == "exif":
                        self.exif_file = candidate_path
                    logger.info(f"✅ Found {file_type}: {candidate}")
                    break
        
        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_count = sum(1 for f in self.images_dir.rglob('*') 
                         if f.suffix.lower() in image_extensions)
        logger.info(f"📸 Found {image_count:,} image files")
        
        return True
    
    def analyze_annotations(self, annotations_df: pd.DataFrame) -> dict:
        """
        Analyze the structure of annotations file
        
        Args:
            annotations_df: DataFrame with annotations
            
        Returns:
            dict: Analysis results
        """
        logger.info("📊 Analyzing annotations structure...")
        
        analysis = {
            'num_samples': len(annotations_df),
            'columns': list(annotations_df.columns),
            'dtypes': annotations_df.dtypes.to_dict(),
            'quality_column': None,
            'image_column': None,
            'quality_stats': None
        }
        
        # Try to identify quality score column
        quality_candidates = [
            'mos', 'quality', 'score', 'rating', 'human_score', 
            'quality_score', 'mos_score', 'mean_opinion_score'
        ]
        
        for col in annotations_df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in quality_candidates):
                if pd.api.types.is_numeric_dtype(annotations_df[col]):
                    analysis['quality_column'] = col
                    analysis['quality_stats'] = {
                        'min': float(annotations_df[col].min()),
                        'max': float(annotations_df[col].max()),
                        'mean': float(annotations_df[col].mean()),
                        'std': float(annotations_df[col].std()),
                        'count': int(annotations_df[col].count())
                    }
                    break
        
        # Try to identify image filename column
        image_candidates = [
            'filename', 'image', 'file', 'path', 'image_name', 
            'image_path', 'image_file', 'img_name'
        ]
        
        for col in annotations_df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in image_candidates):
                analysis['image_column'] = col
                break
        
        # Print analysis
        logger.info(f"📝 Samples: {analysis['num_samples']:,}")
        logger.info(f"📝 Columns: {analysis['columns']}")
        
        if analysis['quality_column']:
            stats = analysis['quality_stats']
            logger.info(f"🎯 Quality column: '{analysis['quality_column']}'")
            logger.info(f"🎯 Quality range: {stats['min']:.2f} - {stats['max']:.2f}")
            logger.info(f"🎯 Quality mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        else:
            logger.warning("❌ Could not identify quality score column")
        
        if analysis['image_column']:
            logger.info(f"🖼️ Image column: '{analysis['image_column']}'")
        else:
            logger.warning("❌ Could not identify image filename column")
        
        return analysis
    
    def create_fusion_annotations(
        self, 
        quality_column: str = None,
        image_column: str = None,
        quality_range: Tuple[float, float] = None
    ) -> bool:
        """
        Create annotations file in fusion training format
        
        Args:
            quality_column: Name of quality score column (auto-detected if None)
            image_column: Name of image filename column (auto-detected if None)
            quality_range: Target quality range for normalization (keeps original if None)
            
        Returns:
            bool: True if successful
        """
        logger.info("🔄 Creating fusion training annotations...")
        
        try:
            # Load main annotations
            annotations_df = pd.read_csv(self.annotations_file)
            analysis = self.analyze_annotations(annotations_df)
            
            # Use provided or auto-detected columns
            quality_col = quality_column or analysis['quality_column']
            image_col = image_column or analysis['image_column']
            
            if not quality_col:
                logger.error("❌ Quality column not specified and could not be auto-detected")
                logger.info("Available columns: " + ", ".join(annotations_df.columns))
                return False
            
            if not image_col:
                logger.error("❌ Image column not specified and could not be auto-detected")
                logger.info("Available columns: " + ", ".join(annotations_df.columns))
                return False
            
            # Extract required columns
            fusion_df = pd.DataFrame()
            fusion_df['image_path'] = annotations_df[image_col]
            fusion_df['human_score'] = annotations_df[quality_col]
            
            # Remove any rows with missing values
            initial_count = len(fusion_df)
            fusion_df = fusion_df.dropna()
            final_count = len(fusion_df)
            
            if initial_count != final_count:
                logger.warning(f"⚠️ Removed {initial_count - final_count} rows with missing values")
            
            # Normalize quality scores if requested
            if quality_range:
                min_target, max_target = quality_range
                min_orig = fusion_df['human_score'].min()
                max_orig = fusion_df['human_score'].max()
                
                # Normalize to [0, 1] then scale to target range
                fusion_df['human_score'] = (fusion_df['human_score'] - min_orig) / (max_orig - min_orig)
                fusion_df['human_score'] = fusion_df['human_score'] * (max_target - min_target) + min_target
                
                logger.info(f"🎯 Normalized quality scores from [{min_orig:.2f}, {max_orig:.2f}] to [{min_target:.2f}, {max_target:.2f}]")
            
            # Validate image paths exist
            logger.info("🔍 Validating image paths...")
            missing_images = []
            valid_rows = []
            
            for idx, row in fusion_df.iterrows():
                image_path = self.images_dir / row['image_path']
                if image_path.exists():
                    valid_rows.append(idx)
                else:
                    missing_images.append(row['image_path'])
            
            if missing_images:
                logger.warning(f"⚠️ Found {len(missing_images)} missing images")
                if len(missing_images) <= 10:
                    for img in missing_images:
                        logger.warning(f"  Missing: {img}")
                else:
                    logger.warning(f"  First 10 missing: {missing_images[:10]}")
                
                fusion_df = fusion_df.loc[valid_rows]
                logger.info(f"✅ Kept {len(fusion_df):,} samples with valid images")
            
            # Keep relative paths for training script compatibility
            # fusion_df['image_path'] remains as filename only
            
            # Save fusion annotations
            output_file = self.output_dir / "annotations_fusion.csv"
            fusion_df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Created fusion annotations: {output_file}")
            logger.info(f"📊 Final dataset: {len(fusion_df):,} samples")
            logger.info(f"🎯 Quality range: {fusion_df['human_score'].min():.2f} - {fusion_df['human_score'].max():.2f}")
            
            # Save analysis report
            analysis_file = self.output_dir / "dataset_analysis.txt"
            with open(analysis_file, 'w') as f:
                f.write("SPAQ Dataset Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Original dataset: {self.spaq_dir}\n")
                f.write(f"Processed dataset: {self.output_dir}\n")
                f.write(f"Processing date: {pd.Timestamp.now()}\n\n")
                
                f.write("Dataset Statistics:\n")
                f.write(f"- Total samples: {len(fusion_df):,}\n")
                f.write(f"- Quality column: {quality_col}\n")
                f.write(f"- Image column: {image_col}\n")
                f.write(f"- Quality range: {fusion_df['human_score'].min():.2f} - {fusion_df['human_score'].max():.2f}\n")
                f.write(f"- Quality mean: {fusion_df['human_score'].mean():.2f}\n")
                f.write(f"- Quality std: {fusion_df['human_score'].std():.2f}\n\n")
                
                if missing_images:
                    f.write(f"Missing images: {len(missing_images)}\n")
                    for img in missing_images[:20]:  # First 20
                        f.write(f"  - {img}\n")
                    if len(missing_images) > 20:
                        f.write(f"  ... and {len(missing_images) - 20} more\n")
            
            logger.info(f"📋 Saved analysis report: {analysis_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating fusion annotations: {e}")
            return False
    
    def process_additional_data(self) -> None:
        """
        Process additional SPAQ data (attributes, scenes, EXIF) for analysis
        """
        logger.info("🔧 Processing additional SPAQ data...")
        
        # Process image attributes
        if self.attributes_file:
            try:
                attrs_df = pd.read_csv(self.attributes_file)
                output_file = self.output_dir / "attributes.csv"
                attrs_df.to_csv(output_file, index=False)
                logger.info(f"✅ Processed attributes: {output_file}")
                logger.info(f"📊 Attributes columns: {list(attrs_df.columns)}")
            except Exception as e:
                logger.error(f"❌ Error processing attributes: {e}")
        
        # Process scene labels
        if self.scenes_file:
            try:
                scenes_df = pd.read_csv(self.scenes_file)
                output_file = self.output_dir / "scenes.csv"
                scenes_df.to_csv(output_file, index=False)
                logger.info(f"✅ Processed scenes: {output_file}")
                logger.info(f"📊 Scene columns: {list(scenes_df.columns)}")
            except Exception as e:
                logger.error(f"❌ Error processing scenes: {e}")
        
        # Process EXIF data
        if self.exif_file:
            try:
                exif_df = pd.read_csv(self.exif_file)
                output_file = self.output_dir / "exif.csv"
                exif_df.to_csv(output_file, index=False)
                logger.info(f"✅ Processed EXIF: {output_file}")
                logger.info(f"📊 EXIF columns: {list(exif_df.columns)}")
            except Exception as e:
                logger.error(f"❌ Error processing EXIF: {e}")
    
    def create_training_splits(
        self, 
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        random_state: int = 42
    ) -> bool:
        """
        Create stratified train/val/test splits
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation  
            test_ratio: Proportion for testing
            random_state: Random seed
            
        Returns:
            bool: True if successful
        """
        logger.info("📂 Creating training splits...")
        
        annotations_file = self.output_dir / "annotations_fusion.csv"
        if not annotations_file.exists():
            logger.error("❌ Fusion annotations not found. Run create_fusion_annotations first.")
            return False
        
        try:
            df = pd.read_csv(annotations_file)
            n_samples = len(df)
            
            # Shuffle data
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Calculate split indices
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)
            
            # Create splits
            train_df = df[:train_end]
            val_df = df[train_end:val_end]
            test_df = df[val_end:]
            
            # Save splits
            splits_dir = self.output_dir / "splits"
            splits_dir.mkdir(exist_ok=True)
            
            train_df.to_csv(splits_dir / "train_annotations.csv", index=False)
            val_df.to_csv(splits_dir / "val_annotations.csv", index=False)
            test_df.to_csv(splits_dir / "test_annotations.csv", index=False)
            
            logger.info(f"✅ Created training splits:")
            logger.info(f"  📚 Train: {len(train_df):,} samples")
            logger.info(f"  🔬 Validation: {len(val_df):,} samples")
            logger.info(f"  🧪 Test: {len(test_df):,} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating splits: {e}")
            return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SPAQ Dataset Preprocessing for Image Quality Fusion')
    
    parser.add_argument('spaq_dir', type=str,
                       help='Path to SPAQ dataset directory')
    parser.add_argument('--output_dir', type=str, default='./processed_spaq',
                       help='Output directory for processed data')
    
    # Column specification
    parser.add_argument('--quality_column', type=str, default=None,
                       help='Name of quality score column (auto-detected if not specified)')
    parser.add_argument('--image_column', type=str, default=None,
                       help='Name of image filename column (auto-detected if not specified)')
    
    # Quality score normalization
    parser.add_argument('--normalize_quality', action='store_true',
                       help='Normalize quality scores to specified range')
    parser.add_argument('--target_range', type=float, nargs=2, default=[1.0, 10.0],
                       help='Target range for quality normalization [min, max]')
    
    # Data splits
    parser.add_argument('--create_splits', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                       help='Test data ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for splits')
    
    return parser.parse_args()


def main():
    """Main preprocessing function"""
    args = parse_args()
    
    print("=" * 60)
    print("SPAQ DATASET PREPROCESSING")
    print("=" * 60)
    print(f"Input: {args.spaq_dir}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Initialize preprocessor
    preprocessor = SPAQPreprocessor(args.spaq_dir, args.output_dir)
    
    # Discover dataset files
    if not preprocessor.discover_files():
        logger.error("❌ Failed to discover required dataset files")
        return 1
    
    # Create fusion annotations
    quality_range = tuple(args.target_range) if args.normalize_quality else None
    
    success = preprocessor.create_fusion_annotations(
        quality_column=args.quality_column,
        image_column=args.image_column,
        quality_range=quality_range
    )
    
    if not success:
        logger.error("❌ Failed to create fusion annotations")
        return 1
    
    # Process additional data
    preprocessor.process_additional_data()
    
    # Create training splits
    if args.create_splits:
        success = preprocessor.create_training_splits(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state
        )
        
        if not success:
            logger.error("❌ Failed to create training splits")
            return 1
    
    print("\n" + "=" * 60)
    print("✅ SPAQ PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    # Print usage instructions
    print("\n🚀 Next steps:")
    print("1. Verify the processed annotations:")
    print(f"   head {args.output_dir}/annotations_fusion.csv")
    print("\n2. Train fusion model:")
    print("   python src/image_quality_fusion/training/train_fusion.py \\")
    print(f"     --image_dir {args.spaq_dir}/images \\")
    print(f"     --annotations {args.output_dir}/annotations_fusion.csv \\")
    print("     --prepare_data \\")
    print("     --model_type deep \\")
    print("     --epochs 100")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)