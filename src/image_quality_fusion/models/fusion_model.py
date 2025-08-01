# src/image_quality_fusion/models/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import json
from pathlib import Path


class ImageQualityFusionModel(nn.Module):
    """
    Fusion model that combines BRISQUE, LAION, and CLIP features
    to predict human-annotated image quality scores.
    
    Architecture:
    - BRISQUE score (1D) -> FC layers
    - LAION score (1D) -> FC layers  
    - CLIP embedding (512D) -> FC layers
    - Concatenate all features -> Final prediction layers
    """
    
    def __init__(
        self,
        clip_embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        output_range: Tuple[float, float] = (1.0, 10.0)
    ):
        """
        Initialize fusion model
        
        Args:
            clip_embed_dim: Dimension of CLIP embeddings
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
            output_range: Expected range of human annotations (min, max)
        """
        super().__init__()
        
        self.clip_embed_dim = clip_embed_dim
        self.hidden_dim = hidden_dim
        self.output_range = output_range
        
        # BRISQUE feature processing (1D -> hidden_dim)
        self.brisque_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, hidden_dim // 4),
            nn.ReLU()
        )
        
        # LAION feature processing (1D -> hidden_dim)
        self.laion_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, hidden_dim // 4),
            nn.ReLU()
        )
        
        # CLIP embedding processing (512D -> hidden_dim)
        self.clip_net = nn.Sequential(
            nn.Linear(clip_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Fusion layers
        fusion_input_dim = (hidden_dim // 4) + (hidden_dim // 4) + (hidden_dim // 2)  # 64 + 64 + 128 = 256
        
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through fusion model
        
        Args:
            features: Dictionary containing:
                - 'brisque': BRISQUE scores [batch_size, 1] or [batch_size]
                - 'laion': LAION aesthetic scores [batch_size, 1] or [batch_size]
                - 'clip': CLIP embeddings [batch_size, 512]
                
        Returns:
            torch.Tensor: Predicted quality scores [batch_size, 1]
        """
        # Ensure correct dimensions
        brisque = features['brisque']
        if brisque.dim() == 1:
            brisque = brisque.unsqueeze(1)  # [batch] -> [batch, 1]
        
        laion = features['laion']
        if laion.dim() == 1:
            laion = laion.unsqueeze(1)  # [batch] -> [batch, 1]
        
        clip = features['clip']
        if clip.dim() == 1:
            clip = clip.unsqueeze(0)  # [512] -> [1, 512] for single sample
        
        # Process individual features
        brisque_features = self.brisque_net(brisque)  # [batch, hidden_dim//4]
        laion_features = self.laion_net(laion)        # [batch, hidden_dim//4]
        clip_features = self.clip_net(clip)           # [batch, hidden_dim//2]
        
        # Concatenate all features
        fused_features = torch.cat([
            brisque_features,
            laion_features, 
            clip_features
        ], dim=1)  # [batch, 256]
        
        # Final prediction
        output = self.fusion_net(fused_features)  # [batch, 1]
        
        # Scale to output range
        min_val, max_val = self.output_range
        output_scaled = min_val + torch.sigmoid(output) * (max_val - min_val)
        
        return output_scaled
    
    def predict(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prediction wrapper (same as forward but explicitly named)"""
        self.eval()
        with torch.no_grad():
            return self.forward(features)
    
    def save_model(self, path: str, metadata: Optional[Dict] = None):
        """
        Save model state and metadata
        
        Args:
            path: Path to save model
            metadata: Optional metadata to save with model
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'clip_embed_dim': self.clip_embed_dim,
                'hidden_dim': self.hidden_dim,
                'output_range': self.output_range
            },
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """
        Load model from saved state
        
        Args:
            path: Path to model file
            device: Device to load model on
            
        Returns:
            Tuple[ImageQualityFusionModel, Dict]: Model and metadata
        """
        save_dict = torch.load(path, map_location=device)
        
        # Extract config
        config = save_dict['model_config']
        
        # Create model
        model = cls(
            clip_embed_dim=config['clip_embed_dim'],
            hidden_dim=config['hidden_dim'],
            output_range=tuple(config['output_range'])
        )
        
        # Load state
        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        
        return model, save_dict.get('metadata', {})


class WeightedFusionModel(nn.Module):
    """
    Alternative fusion model using learned weighted combination
    """
    
    def __init__(self, output_range: Tuple[float, float] = (1.0, 10.0)):
        super().__init__()
        self.output_range = output_range
        
        # Learnable weights for each component
        self.brisque_weight = nn.Parameter(torch.tensor(1.0))
        self.laion_weight = nn.Parameter(torch.tensor(1.0))
        self.clip_weight = nn.Parameter(torch.tensor(1.0))
        
        # CLIP embedding dimension reduction
        self.clip_reducer = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalize to 0-1
        )
        
        # Final combination layer
        self.final_layer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Reduce CLIP embedding to single score
        clip_score = self.clip_reducer(features['clip'])
        
        # Weight and combine scores
        weighted_features = torch.cat([
            features['brisque'] * self.brisque_weight,
            features['laion'] * self.laion_weight,
            clip_score * self.clip_weight
        ], dim=1)
        
        # Final prediction
        output = self.final_layer(weighted_features)
        
        # Scale to output range
        min_val, max_val = self.output_range
        output_scaled = min_val + torch.sigmoid(output) * (max_val - min_val)
        
        return output_scaled


class EnsembleFusionModel(nn.Module):
    """
    Ensemble of multiple fusion approaches
    """
    
    def __init__(self, output_range: Tuple[float, float] = (1.0, 10.0)):
        super().__init__()
        self.output_range = output_range
        
        # Multiple fusion models
        self.deep_fusion = ImageQualityFusionModel(output_range=output_range)
        self.weighted_fusion = WeightedFusionModel(output_range=output_range)
        
        # Ensemble combination
        self.ensemble_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get predictions from both models
        deep_pred = self.deep_fusion(features)
        weighted_pred = self.weighted_fusion(features)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_pred = weights[0] * deep_pred + weights[1] * weighted_pred
        
        return ensemble_pred