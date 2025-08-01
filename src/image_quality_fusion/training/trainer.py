# src/image_quality_fusion/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime

from ..models.fusion_model import ImageQualityFusionModel, WeightedFusionModel, EnsembleFusionModel
from ..data.preprocessing import ImageQualityExtractor


class ImageQualityDataset(Dataset):
    """
    Dataset for image quality fusion model training
    """
    
    def __init__(
        self,
        annotations_path: str,
        features_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        extractor: Optional[ImageQualityExtractor] = None,
        image_dir: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            annotations_path: Path to CSV with image_path, human_score columns
            features_path: Path to precomputed features CSV (optional)
            embeddings_path: Path to precomputed embeddings .npy file (optional)
            extractor: ImageQualityExtractor for on-the-fly feature extraction
            image_dir: Directory containing images (if using relative paths)
        """
        self.annotations = pd.read_csv(annotations_path)
        self.image_dir = Path(image_dir) if image_dir else None
        self.extractor = extractor
        
        # Load precomputed features if available
        if features_path and Path(features_path).exists():
            self.features_df = pd.read_csv(features_path)
            if embeddings_path and Path(embeddings_path).exists():
                self.embeddings = np.load(embeddings_path)
            else:
                self.embeddings = None
            self.precomputed = True
        else:
            self.features_df = None
            self.embeddings = None
            self.precomputed = False
            
            if not extractor:
                raise ValueError("Must provide either precomputed features or extractor")
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate dataset integrity"""
        required_cols = ['image_path', 'human_score']
        for col in required_cols:
            if col not in self.annotations.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for NaN values
        if self.annotations['human_score'].isna().any():
            print("Warning: Found NaN values in human_score, will be filtered out")
            self.annotations = self.annotations.dropna(subset=['human_score'])
        
        print(f"Dataset loaded: {len(self.annotations)} samples")
        print(f"Human score range: {self.annotations['human_score'].min():.2f} - {self.annotations['human_score'].max():.2f}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample by index
        
        Returns:
            Tuple of (features_dict, human_score)
        """
        row = self.annotations.iloc[idx]
        image_path = row['image_path']
        human_score = float(row['human_score'])
        
        # Get image features
        if self.precomputed:
            features = self._get_precomputed_features(image_path, idx)
        else:
            features = self._extract_features_on_demand(image_path)
        
        # Convert to tensors
        features_tensor = {
            'brisque': torch.tensor([features['brisque_normalized']], dtype=torch.float32),
            'laion': torch.tensor([features['aesthetic_normalized']], dtype=torch.float32),
            'clip': torch.tensor(features['clip_embedding'], dtype=torch.float32)
        }
        
        human_score_tensor = torch.tensor([human_score], dtype=torch.float32)
        
        return features_tensor, human_score_tensor
    
    def _get_precomputed_features(self, image_path: str, idx: int) -> Dict:
        """Get features from precomputed files"""
        # Find matching row in features
        feature_row = self.features_df[self.features_df['image_path'] == image_path]
        if feature_row.empty:
            raise ValueError(f"No precomputed features found for {image_path}")
        
        feature_row = feature_row.iloc[0]
        
        features = {
            'brisque_normalized': feature_row['brisque_normalized'],
            'aesthetic_normalized': feature_row['aesthetic_normalized'],
            'clip_embedding': self.embeddings[idx] if self.embeddings is not None else np.zeros(512)
        }
        
        return features
    
    def _extract_features_on_demand(self, image_path: str) -> Dict:
        """Extract features on-demand using extractor"""
        # Handle relative paths
        if self.image_dir and not Path(image_path).is_absolute():
            full_path = self.image_dir / image_path
        else:
            full_path = Path(image_path)
        
        features = self.extractor.extract_features_single_image(str(full_path))
        if features is None:
            raise ValueError(f"Failed to extract features for {image_path}")
        
        return features


class FusionModelTrainer:
    """
    Trainer for image quality fusion models
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        output_dir: str = './outputs'
    ):
        """
        Initialize trainer
        
        Args:
            model: Fusion model to train
            device: Device to train on ('auto', 'cpu', 'cuda', 'mps')
            output_dir: Directory to save outputs
        """
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
    def _setup_device(self, device: str) -> str:
        """Setup training device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        save_best: bool = True,
        scheduler_params: Optional[Dict] = None,
        num_workers: int = None,
        mixed_precision: bool = False
    ) -> Dict:
        """
        Train the fusion model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            save_best: Whether to save best model
            scheduler_params: Learning rate scheduler parameters
            
        Returns:
            Dict: Training history and metrics
        """
        # Create optimized data loaders
        train_loader = self._create_optimized_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            is_training=True,
            num_workers=num_workers
        )
        
        val_loader = None
        if val_dataset:
            val_loader = self._create_optimized_dataloader(
                val_dataset,
                batch_size=batch_size,
                is_training=False,
                num_workers=num_workers
            )
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        criterion = nn.MSELoss()
        
        # Setup mixed precision training
        scaler = None
        if mixed_precision:
            if self.device == 'cuda':
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                self.logger.info("Using CUDA mixed precision training")
            elif self.device == 'mps':
                # MPS doesn't support GradScaler yet, so we'll use torch.autocast only
                self.logger.info("Using MPS autocast (no gradient scaling)")
            else:
                self.logger.warning("Mixed precision requested but not supported on CPU")
                mixed_precision = False
        
        # Setup scheduler
        scheduler = None
        if scheduler_params:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_params
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, criterion, scaler, mixed_precision)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = None
            val_metrics = {}
            if val_loader:
                val_loss, val_metrics = self._validate_epoch(val_loader, criterion)
                self.val_losses.append(val_loss)
                self.metrics_history.append(val_metrics)
            
            # Logging
            log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}"
                log_msg += f", Val MAE: {val_metrics.get('mae', 0):.4f}"
                log_msg += f", Val R²: {val_metrics.get('r2', 0):.4f}"
            
            self.logger.info(log_msg)
            
            # Learning rate scheduling
            if scheduler and val_loss is not None:
                scheduler.step(val_loss)
            
            # Early stopping and model saving
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_best:
                        self._save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_loss, val_metrics, is_best=False)
        
        # Final evaluation
        final_metrics = {}
        if val_loader:
            final_metrics = self._final_evaluation(val_loader)
        
        # Save training plots
        self._save_training_plots()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, scaler=None, mixed_precision: bool = False) -> float:
        """Train for one epoch with optional mixed precision"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_features, batch_targets in progress_bar:
            # Move to device
            batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            
            if mixed_precision and self.device == 'cuda' and scaler is not None:
                # CUDA mixed precision with gradient scaling
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch_features)
                    loss = criterion(predictions, batch_targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            elif mixed_precision and self.device == 'mps':
                # MPS autocast (no gradient scaling available)
                with torch.autocast(device_type='cpu', dtype=torch.float16):
                    predictions = self.model(batch_features)
                    loss = criterion(predictions, batch_targets)
                
                loss.backward()
                optimizer.step()
                
            else:
                # Standard precision training
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_targets)
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Memory cleanup every 10 batches
            if (len(progress_bar) % 10) == 0:
                if self.device == 'mps':
                    torch.mps.empty_cache()
                elif self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                # Move to device
                batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        
        metrics = {
            'mse': mean_squared_error(all_targets, all_predictions),
            'mae': mean_absolute_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions)
        }
        
        return total_loss / len(val_loader), metrics
    
    def _final_evaluation(self, val_loader: DataLoader) -> Dict:
        """Perform final detailed evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = {k: v.to(self.device) for k, v in batch_features.items()}
                predictions = self.model(batch_features)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # Comprehensive metrics
        metrics = {
            'mse': mean_squared_error(all_targets, all_predictions),
            'rmse': np.sqrt(mean_squared_error(all_targets, all_predictions)),
            'mae': mean_absolute_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions),
            'correlation': np.corrcoef(all_targets, all_predictions)[0, 1]
        }
        
        self.logger.info("Final Evaluation Metrics:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name.upper()}: {value:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, val_loss: Optional[float], metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        suffix = "best" if is_best else f"epoch_{epoch+1}"
        checkpoint_path = self.output_dir / f"model_{suffix}.pth"
        
        metadata = {
            'epoch': epoch,
            'val_loss': val_loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.model.save_model(str(checkpoint_path), metadata)
    
    def _save_training_plots(self):
        """Save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Metrics over time
        if self.metrics_history:
            epochs = range(len(self.metrics_history))
            axes[0, 1].plot(epochs, [m.get('mae', 0) for m in self.metrics_history], label='MAE')
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(epochs, [m.get('r2', 0) for m in self.metrics_history], label='R²')
            axes[1, 0].set_title('R² Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Training Complete', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('Training Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to {self.output_dir / 'training_plots.png'}")
    
    def _create_optimized_dataloader(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        is_training: bool = True,
        num_workers: int = None
    ) -> DataLoader:
        """
        Create optimized DataLoader for M1 MacBook Pro
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            is_training: Whether this is for training (affects shuffle, drop_last)
            num_workers: Number of workers (auto-calculated if None)
            
        Returns:
            DataLoader: Optimized data loader
        """
        import os
        
        # Calculate optimal workers for M1 MacBook Pro (8-core + 2 efficiency)
        if num_workers is None:
            num_workers = min(8, max(1, os.cpu_count() - 2))
        
        # Conservative batch size to prevent memory issues
        effective_batch_size = batch_size
        if hasattr(dataset, 'precomputed') and dataset.precomputed:
            # Even with cached features, be conservative with large datasets
            if len(dataset) > 10000:  # Large dataset
                effective_batch_size = min(batch_size, 256)
                self.logger.info(f"Large dataset detected, using conservative batch size {effective_batch_size}")
            else:
                effective_batch_size = min(batch_size * 2, 512)
                self.logger.info(f"Using larger batch size {effective_batch_size} for cached features")
        
        # Check available memory and adjust if needed
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < 8:  # Less than 8GB available
                effective_batch_size = min(effective_batch_size, 256)
                num_workers = min(num_workers, 4)
                self.logger.warning(f"Limited memory detected, reducing batch size to {effective_batch_size}")
        except ImportError:
            pass
        
        self.logger.info(f"DataLoader config: batch_size={effective_batch_size}, num_workers={num_workers}, device={self.device}")
        
        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=self.device in ['mps', 'cuda'],
            prefetch_factor=4 if num_workers > 0 else 2,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=is_training  # Drop incomplete batches in training only
        )


def create_train_val_split(
    annotations_path: str,
    train_ratio: float = 0.8,
    random_state: int = 42,
    stratify_by: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split from annotations
    
    Args:
        annotations_path: Path to annotations CSV
        train_ratio: Ratio of data for training
        random_state: Random seed
        stratify_by: Column to stratify split by (optional)
        
    Returns:
        Tuple of (train_df, val_df)
    """
    df = pd.read_csv(annotations_path)
    
    if stratify_by and stratify_by in df.columns:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, 
            train_size=train_ratio,
            random_state=random_state,
            stratify=df[stratify_by]
        )
    else:
        # Simple random split
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * train_ratio)
        train_df = df_shuffled[:split_idx]
        val_df = df_shuffled[split_idx:]
    
    print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation")
    return train_df, val_df