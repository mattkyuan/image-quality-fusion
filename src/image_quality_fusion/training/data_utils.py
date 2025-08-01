# src/image_quality_fusion/training/data_utils.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from ..data.preprocessing import ImageQualityExtractor


def prepare_training_data(
    image_dir: str,
    annotations_path: str,
    output_dir: str = './training_data',
    batch_size: int = 500,
    force_recompute: bool = False
) -> Tuple[str, str]:
    """
    Prepare training data by extracting features for all images
    
    Args:
        image_dir: Directory containing images
        annotations_path: Path to CSV with image_path, human_score columns
        output_dir: Directory to save processed features
        batch_size: Batch size for feature extraction
        force_recompute: Whether to recompute existing features
        
    Returns:
        Tuple of (features_path, embeddings_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    features_path = output_dir / 'features.csv'
    embeddings_path = output_dir / 'embeddings.npy'
    
    # Check if already exists
    if not force_recompute and features_path.exists() and embeddings_path.exists():
        print(f"Features already exist at {features_path}")
        return str(features_path), str(embeddings_path)
    
    # Load annotations
    annotations = pd.read_csv(annotations_path)
    print(f"Processing {len(annotations)} images...")
    
    # Initialize extractor with M1 optimizations
    extractor = ImageQualityExtractor()
    
    # Prepare image paths
    image_paths = []
    for _, row in annotations.iterrows():
        image_path = row['image_path']
        
        # Handle relative paths
        if not Path(image_path).is_absolute():
            full_path = Path(image_dir) / image_path
        else:
            full_path = Path(image_path)
        
        image_paths.append(str(full_path))
    
    # Use optimized batch processing
    print("Using M1-optimized batch processing...")
    results_df = extractor.extract_features_batch(
        image_paths, 
        output_path=str(features_path),
        batch_size=batch_size
    )
    
    # Features and embeddings are already saved by extract_features_batch
    print(f"Successfully processed {len(results_df)} images")
    
    # Check for failed images
    failed_count = len(image_paths) - len(results_df)
    if failed_count > 0:
        print(f"Failed to process {failed_count} images")
    
    # Features and embeddings are already saved by extract_features_batch
    print(f"M1-optimized feature extraction completed!")
    print(f"Processed {len(results_df)} images successfully")
    
    return str(features_path), str(embeddings_path)


def analyze_annotations(annotations_path: str, output_dir: Optional[str] = None) -> Dict:
    """
    Analyze human annotation distribution and statistics
    
    Args:
        annotations_path: Path to annotations CSV
        output_dir: Directory to save analysis plots (optional)
        
    Returns:
        Dict: Analysis results
    """
    df = pd.read_csv(annotations_path)
    
    # Basic statistics
    stats = {
        'count': len(df),
        'mean': df['human_score'].mean(),
        'std': df['human_score'].std(),
        'min': df['human_score'].min(),
        'max': df['human_score'].max(),
        'median': df['human_score'].median(),
        'q25': df['human_score'].quantile(0.25),
        'q75': df['human_score'].quantile(0.75)
    }
    
    print("Human Annotation Statistics:")
    print(f"  Count: {stats['count']:,}")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  IQR: [{stats['q25']:.3f}, {stats['q75']:.3f}]")
    
    # Create visualizations if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(df['human_score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Human Scores')
        axes[0, 0].set_xlabel('Human Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(df['human_score'])
        axes[0, 1].set_title('Human Score Box Plot')
        axes[0, 1].set_ylabel('Human Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_scores = np.sort(df['human_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 0].plot(sorted_scores, cumulative)
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].set_xlabel('Human Score')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Score density by range
        score_ranges = pd.cut(df['human_score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        range_counts = score_ranges.value_counts()
        axes[1, 1].pie(range_counts.values, labels=range_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Score Distribution by Range')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'annotation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plots saved to: {output_dir / 'annotation_analysis.png'}")
    
    return stats


def create_stratified_split(
    annotations_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    score_bins: int = 5,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Create stratified train/val/test splits based on human scores
    
    Args:
        annotations_path: Path to annotations CSV
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        score_bins: Number of bins for stratification
        random_state: Random seed
        
    Returns:
        Dict: Paths to split files
    """
    df = pd.read_csv(annotations_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create score bins for stratification
    df['score_bin'] = pd.cut(df['human_score'], bins=score_bins, labels=False)
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['score_bin'],
        random_state=random_state
    )
    
    # Second split: separate train and validation
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        stratify=train_val_df['score_bin'],
        random_state=random_state
    )
    
    # Remove the temporary score_bin column
    train_df = train_df.drop('score_bin', axis=1)
    val_df = val_df.drop('score_bin', axis=1)
    test_df = test_df.drop('score_bin', axis=1)
    
    # Save splits
    split_paths = {
        'train': str(output_dir / 'train_annotations.csv'),
        'val': str(output_dir / 'val_annotations.csv'),
        'test': str(output_dir / 'test_annotations.csv')
    }
    
    train_df.to_csv(split_paths['train'], index=False)
    val_df.to_csv(split_paths['val'], index=False)
    test_df.to_csv(split_paths['test'], index=False)
    
    print(f"Dataset splits created:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return split_paths


def validate_training_data(
    annotations_path: str,
    features_path: str,
    embeddings_path: str,
    image_dir: Optional[str] = None
) -> bool:
    """
    Validate that training data is consistent and ready for training
    
    Args:
        annotations_path: Path to annotations CSV
        features_path: Path to features CSV
        embeddings_path: Path to embeddings .npy file
        image_dir: Directory containing images (optional, for file existence check)
        
    Returns:
        bool: True if validation passes
    """
    print("Validating training data...")
    
    # Load data
    annotations = pd.read_csv(annotations_path)
    features = pd.read_csv(features_path)
    embeddings = np.load(embeddings_path)
    
    issues = []
    
    # Check row counts match
    if len(annotations) != len(features):
        issues.append(f"Annotation count ({len(annotations)}) != features count ({len(features)})")
    
    if len(features) != len(embeddings):
        issues.append(f"Features count ({len(features)}) != embeddings count ({len(embeddings)})")
    
    # Check required columns
    required_annotation_cols = ['image_path', 'human_score']
    for col in required_annotation_cols:
        if col not in annotations.columns:
            issues.append(f"Missing column in annotations: {col}")
    
    required_feature_cols = ['image_path', 'brisque_normalized', 'aesthetic_normalized']
    for col in required_feature_cols:
        if col not in features.columns:
            issues.append(f"Missing column in features: {col}")
    
    # Check for NaN values
    if annotations['human_score'].isna().any():
        issues.append("Found NaN values in human_score")
    
    if features[['brisque_normalized', 'aesthetic_normalized']].isna().any().any():
        issues.append("Found NaN values in normalized features")
    
    # Check data ranges
    if not (0 <= features['brisque_normalized']).all() or not (features['brisque_normalized'] <= 1).all():
        issues.append("BRISQUE normalized scores not in [0, 1] range")
    
    if not (0 <= features['aesthetic_normalized']).all() or not (features['aesthetic_normalized'] <= 1).all():
        issues.append("Aesthetic normalized scores not in [0, 1] range")
    
    # Check image path consistency
    annotation_paths = set(annotations['image_path'])
    feature_paths = set(features['image_path'])
    
    missing_in_features = annotation_paths - feature_paths
    if missing_in_features:
        issues.append(f"Images in annotations but not in features: {len(missing_in_features)}")
    
    extra_in_features = feature_paths - annotation_paths
    if extra_in_features:
        issues.append(f"Images in features but not in annotations: {len(extra_in_features)}")
    
    # Check embedding dimensions
    expected_dim = 512  # Standard CLIP dimension
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        issues.append(f"Expected embeddings shape (N, {expected_dim}), got {embeddings.shape}")
    
    # Check for infinite or very large values
    if not np.isfinite(embeddings).all():
        issues.append("Found non-finite values in embeddings")
    
    # Optionally check file existence
    if image_dir:
        image_dir = Path(image_dir)
        missing_files = []
        for img_path in annotations['image_path'].iloc[:10]:  # Check first 10 as sample
            if not Path(img_path).is_absolute():
                full_path = image_dir / img_path
            else:
                full_path = Path(img_path)
            
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            issues.append(f"Sample missing image files: {missing_files[:3]}...")
    
    # Report results
    if issues:
        print("❌ Validation FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Validation PASSED:")
        print(f"  - {len(annotations):,} samples ready for training")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Human score range: [{annotations['human_score'].min():.2f}, {annotations['human_score'].max():.2f}]")
        return True


def create_sample_annotations(
    image_dir: str,
    output_path: str,
    num_samples: int = 100,
    score_range: Tuple[float, float] = (1.0, 10.0),
    random_state: int = 42
) -> str:
    """
    Create sample annotations file for testing (with synthetic scores)
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save annotations CSV
        num_samples: Number of samples to create
        score_range: Range of synthetic scores
        random_state: Random seed
        
    Returns:
        str: Path to created annotations file
    """
    np.random.seed(random_state)
    
    # Find image files
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'**/*{ext}'))
        image_files.extend(image_dir.glob(f'**/*{ext.upper()}'))
    
    if len(image_files) < num_samples:
        print(f"Warning: Only found {len(image_files)} images, using all of them")
        num_samples = len(image_files)
    
    # Sample images
    selected_images = np.random.choice(image_files, size=num_samples, replace=False)
    
    # Generate synthetic scores
    min_score, max_score = score_range
    synthetic_scores = np.random.uniform(min_score, max_score, size=num_samples)
    
    # Create annotations DataFrame
    annotations = pd.DataFrame({
        'image_path': [str(img.relative_to(image_dir)) for img in selected_images],
        'human_score': synthetic_scores
    })
    
    # Save annotations
    annotations.to_csv(output_path, index=False)
    
    print(f"Created sample annotations: {output_path}")
    print(f"  - {len(annotations)} samples")
    print(f"  - Score range: [{synthetic_scores.min():.2f}, {synthetic_scores.max():.2f}]")
    
    return output_path