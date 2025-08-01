#!/usr/bin/env python3
# src/image_quality_fusion/training/evaluate.py
"""
Evaluation and inference script for trained fusion models
"""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.image_quality_fusion.models.fusion_model import ImageQualityFusionModel
from src.image_quality_fusion.training.trainer import ImageQualityDataset
from src.image_quality_fusion.data.preprocessing import ImageQualityExtractor


class FusionModelEvaluator:
    """Evaluator for trained fusion models"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model
            device: Device to run inference on
        """
        self.device = self._setup_device(device)
        self.model, self.metadata = ImageQualityFusionModel.load_model(model_path, self.device)
        self.model.eval()
        
        print(f"Loaded model from: {model_path}")
        print(f"Device: {self.device}")
        if 'epoch' in self.metadata:
            print(f"Model trained for {self.metadata['epoch'] + 1} epochs")
    
    def _setup_device(self, device: str) -> str:
        """Setup inference device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def evaluate_dataset(
        self, 
        dataset: ImageQualityDataset,
        batch_size: int = 32,
        save_predictions: bool = True,
        output_path: str = None
    ) -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for inference
            save_predictions: Whether to save predictions
            output_path: Path to save predictions CSV
            
        Returns:
            Dict: Evaluation metrics and predictions
        """
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        all_predictions = []
        all_targets = []
        all_image_paths = []
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        with torch.no_grad():
            for i, (batch_features, batch_targets) in enumerate(dataloader):
                # Move to device
                batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
                
                # Forward pass
                predictions = self.model(batch_features)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch_targets.numpy().flatten())
                
                # Get image paths for this batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                for idx in range(start_idx, end_idx):
                    image_path = dataset.annotations.iloc[idx]['image_path']
                    all_image_paths.append(image_path)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'image_path': all_image_paths,
            'human_score': targets,
            'predicted_score': predictions,
            'absolute_error': np.abs(targets - predictions),
            'squared_error': (targets - predictions) ** 2
        })
        
        # Save predictions if requested
        if save_predictions and output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
        
        return {
            'metrics': metrics,
            'predictions': results_df,
            'raw_predictions': predictions,
            'raw_targets': targets
        }
    
    def predict_single_image(
        self, 
        image_path: str, 
        extractor: ImageQualityExtractor = None
    ) -> float:
        """
        Predict quality score for a single image
        
        Args:
            image_path: Path to image
            extractor: Feature extractor (created if None)
            
        Returns:
            float: Predicted quality score
        """
        if extractor is None:
            extractor = ImageQualityExtractor()
        
        # Extract features
        features = extractor.extract_features_single_image(image_path)
        if features is None:
            raise ValueError(f"Failed to extract features for {image_path}")
        
        # Convert to tensors
        features_tensor = {
            'brisque': torch.tensor([features['brisque_normalized']], dtype=torch.float32).to(self.device),
            'laion': torch.tensor([features['aesthetic_normalized']], dtype=torch.float32).to(self.device),
            'clip': torch.tensor(features['clip_embedding'], dtype=torch.float32).to(self.device)
        }
        
        # Predict
        with torch.no_grad():
            prediction = self.model(features_tensor)
            return float(prediction.cpu().numpy()[0, 0])
    
    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            # Regression metrics
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            
            # Correlation metrics
            'pearson_r': pearsonr(targets, predictions)[0],
            'spearman_r': spearmanr(targets, predictions)[0],
            
            # Additional metrics
            'mean_target': np.mean(targets),
            'mean_prediction': np.mean(predictions),
            'std_target': np.std(targets),
            'std_prediction': np.std(predictions),
            
            # Error analysis
            'mean_absolute_error': np.mean(np.abs(targets - predictions)),
            'median_absolute_error': np.median(np.abs(targets - predictions)),
            'max_absolute_error': np.max(np.abs(targets - predictions)),
            
            # Relative metrics
            'mape': np.mean(np.abs((targets - predictions) / targets)) * 100,  # Mean Absolute Percentage Error
        }
        
        return metrics
    
    def create_evaluation_plots(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray,
        output_dir: str
    ):
        """Create comprehensive evaluation plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Scatter plot: Predicted vs Actual
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=20)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Human Score (Ground Truth)')
        axes[0, 0].set_ylabel('Predicted Score')
        axes[0, 0].set_title('Predicted vs Actual Scores')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Calculate and add R² to plot
        r2 = r2_score(targets, predictions)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Residual plot
        residuals = predictions - targets
        axes[0, 1].scatter(targets, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Human Score (Ground Truth)')
        axes[0, 1].set_ylabel('Residuals (Predicted - Actual)')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        absolute_errors = np.abs(residuals)
        axes[0, 2].hist(absolute_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Absolute Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Absolute Errors')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add median and mean error lines
        mean_error = np.mean(absolute_errors)
        median_error = np.median(absolute_errors)
        axes[0, 2].axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.3f}')
        axes[0, 2].axvline(median_error, color='blue', linestyle='--', label=f'Median: {median_error:.3f}')
        axes[0, 2].legend()
        
        # 4. QQ plot for normality of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot: Residuals vs Normal Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Bland-Altman plot
        mean_scores = (targets + predictions) / 2
        diff_scores = predictions - targets
        axes[1, 1].scatter(mean_scores, diff_scores, alpha=0.6, s=20)
        
        mean_diff = np.mean(diff_scores)
        std_diff = np.std(diff_scores)
        axes[1, 1].axhline(mean_diff, color='red', linestyle='-', label=f'Mean: {mean_diff:.3f}')
        axes[1, 1].axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', alpha=0.7, label=f'+1.96σ: {mean_diff + 1.96*std_diff:.3f}')
        axes[1, 1].axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', alpha=0.7, label=f'-1.96σ: {mean_diff - 1.96*std_diff:.3f}')
        
        axes[1, 1].set_xlabel('Mean Score ((Human + Predicted) / 2)')
        axes[1, 1].set_ylabel('Difference (Predicted - Human)')
        axes[1, 1].set_title('Bland-Altman Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Score distribution comparison
        axes[1, 2].hist(targets, bins=30, alpha=0.5, label='Human Scores', color='blue', edgecolor='black')
        axes[1, 2].hist(predictions, bins=30, alpha=0.5, label='Predicted Scores', color='red', edgecolor='black')
        axes[1, 2].set_xlabel('Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Score Distribution Comparison')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to: {plot_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Fusion Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test_annotations', type=str, required=True,
                       help='Path to test annotations CSV')
    parser.add_argument('--features_path', type=str, required=True,
                       help='Path to features CSV')
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to embeddings .npy file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for inference')
    
    # Single image prediction
    parser.add_argument('--predict_single', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Base directory for images (for single prediction)')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("=" * 60)
    print("IMAGE QUALITY FUSION MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_annotations}")
    
    # Create evaluator
    evaluator = FusionModelEvaluator(args.model_path, args.device)
    
    # Handle single image prediction
    if args.predict_single:
        print(f"\n🔍 Predicting single image: {args.predict_single}")
        try:
            score = evaluator.predict_single_image(args.predict_single)
            print(f"Predicted quality score: {score:.3f}")
        except Exception as e:
            print(f"Error predicting single image: {e}")
        return 0
    
    # Create test dataset
    print("\n📦 Loading test dataset...")
    test_dataset = ImageQualityDataset(
        annotations_path=args.test_annotations,
        features_path=args.features_path,
        embeddings_path=args.embeddings_path
    )
    print(f"Test samples: {len(test_dataset):,}")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluator.evaluate_dataset(
        dataset=test_dataset,
        batch_size=args.batch_size,
        save_predictions=True,
        output_path=str(output_dir / 'predictions.csv')
    )
    
    # Print metrics
    print("\n📈 Evaluation Results:")
    print("-" * 40)
    metrics = results['metrics']
    
    key_metrics = ['rmse', 'mae', 'r2', 'pearson_r', 'spearman_r']
    for metric in key_metrics:
        if metric in metrics:
            print(f"{metric.upper():>12}: {metrics[metric]:.4f}")
    
    print(f"\nPrediction Range: [{results['raw_predictions'].min():.2f}, {results['raw_predictions'].max():.2f}]")
    print(f"Target Range:     [{results['raw_targets'].min():.2f}, {results['raw_targets'].max():.2f}]")
    
    # Create evaluation plots
    print("\n📊 Creating evaluation plots...")
    evaluator.create_evaluation_plots(
        targets=results['raw_targets'],
        predictions=results['raw_predictions'],
        output_dir=str(output_dir)
    )
    
    # Save detailed metrics
    metrics_path = output_dir / 'evaluation_metrics.json'
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                json_metrics[k] = v.item()
            else:
                json_metrics[k] = float(v)
        
        json.dump(json_metrics, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)