#!/usr/bin/env python3
# src/image_quality_fusion/training/train_fusion.py
"""
Training script for image quality fusion models
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# Use relative imports within the package
from ..models.fusion_model import (
    ImageQualityFusionModel, 
    WeightedFusionModel, 
    EnsembleFusionModel
)
from .trainer import (
    FusionModelTrainer,
    ImageQualityDataset
)
from .data_utils import (
    prepare_training_data,
    analyze_annotations,
    create_stratified_split,
    validate_training_data
)
from ..utils.paths import get_project_root, resolve_path_from_project_root


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Image Quality Fusion Model')
    
    # Data arguments
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotations CSV (image_path, human_score)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    
    # Data preparation
    parser.add_argument('--prepare_data', action='store_true',
                       help='Extract features before training')
    parser.add_argument('--features_dir', type=str, default='./training_data',
                       help='Directory for extracted features')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Recompute features even if they exist')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='deep',
                       choices=['deep', 'weighted', 'ensemble'],
                       help='Type of fusion model')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--output_range', type=float, nargs=2, default=[1.0, 10.0],
                       help='Expected range of human scores [min, max]')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (conservative for M1 with 32GB RAM)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Data splitting
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                       help='Ratio of data for testing')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of workers for data loading (auto-detected for M1 if None)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training for faster training')
    
    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--tags', type=str, nargs='*', default=[],
                       help='Tags for this experiment')
    
    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directory and configuration"""
    output_dir = Path(args.output_dir)
    
    if args.experiment_name:
        output_dir = output_dir / args.experiment_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment directory: {output_dir}")
    return output_dir


def create_model(args):
    """Create fusion model based on arguments"""
    output_range = tuple(args.output_range)
    
    if args.model_type == 'deep':
        model = ImageQualityFusionModel(
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            output_range=output_range
        )
    elif args.model_type == 'weighted':
        model = WeightedFusionModel(output_range=output_range)
    elif args.model_type == 'ensemble':
        model = EnsembleFusionModel(output_range=output_range)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Created {args.model_type} fusion model")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup experiment
    output_dir = setup_experiment(args)
    
    print("=" * 60)
    print("IMAGE QUALITY FUSION MODEL TRAINING")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Images: {args.image_dir}")
    print(f"Annotations: {args.annotations}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Analyze annotations
    print("📊 Analyzing annotations...")
    analyze_annotations(args.annotations, output_dir / 'analysis')
    
    # Step 2: Prepare training data
    if args.prepare_data:
        print("\n🔧 Preparing training data...")
        features_path, embeddings_path = prepare_training_data(
            image_dir=args.image_dir,
            annotations_path=args.annotations,
            output_dir=args.features_dir,
            force_recompute=args.force_recompute
        )
    else:
        # Use existing features
        features_dir = Path(args.features_dir)
        features_path = str(features_dir / 'features.csv')
        embeddings_path = str(features_dir / 'embeddings.npy')
        
        if not Path(features_path).exists() or not Path(embeddings_path).exists():
            raise FileNotFoundError(
                f"Features not found at {features_path} or {embeddings_path}. "
                "Use --prepare_data to extract features."
            )
    
    # Step 3: Validate training data
    print("\n✅ Validating training data...")
    if not validate_training_data(
        args.annotations, features_path, embeddings_path, args.image_dir
    ):
        print("❌ Data validation failed. Please fix issues before training.")
        return 1
    
    # Step 4: Create train/val splits
    print("\n📂 Creating dataset splits...")
    split_paths = create_stratified_split(
        annotations_path=args.annotations,
        output_dir=output_dir / 'splits',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    # Step 5: Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = ImageQualityDataset(
        annotations_path=split_paths['train'],
        features_path=features_path,
        embeddings_path=embeddings_path
    )
    
    val_dataset = ImageQualityDataset(
        annotations_path=split_paths['val'],
        features_path=features_path,
        embeddings_path=embeddings_path
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Step 6: Create model
    print(f"\n🧠 Creating {args.model_type} fusion model...")
    model = create_model(args)
    
    # Step 7: Create trainer
    print(f"\n🏋️ Setting up trainer...")
    trainer = FusionModelTrainer(
        model=model,
        device=args.device,
        output_dir=str(output_dir)
    )
    
    # Step 8: Train model
    print("\n🚀 Starting training...")
    scheduler_params = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 5
    }
    
    print(f"Training configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Workers: {args.num_workers or 'auto-detected'}")
    print(f"  - Mixed precision: {args.mixed_precision}")
    print(f"  - Device: {args.device}")
    
    training_history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        scheduler_params=scheduler_params,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision
    )
    
    # Step 9: Save final results
    print("\n💾 Saving final results...")
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {}
        for key, value in training_history.items():
            if isinstance(value, list):
                serializable_history[key] = [float(x) if hasattr(x, 'item') else x for x in value]
            elif isinstance(value, dict):
                serializable_history[key] = {k: float(v) if hasattr(v, 'item') else v for k, v in value.items()}
            else:
                serializable_history[key] = float(value) if hasattr(value, 'item') else value
        
        json.dump(serializable_history, f, indent=2)
    
    print(f"Training complete! Results saved to: {output_dir}")
    print("\nFinal Metrics:")
    if 'final_metrics' in training_history:
        for metric, value in training_history['final_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)