"""
Hugging Face Hub wrapper for Image Quality Fusion model
"""
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image
import numpy as np
from typing import Dict, Union, Optional
import json

from .fusion_model import ImageQualityFusionModel
from ..data.preprocessing import ImageQualityExtractor


class ImageQualityFusionHF(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/yourusername/image-quality-fusion",
    tags=["image-quality", "brisque", "aesthetic", "clip", "fusion"],
    license="mit",
):
    """
    Hugging Face compatible wrapper for Image Quality Fusion model.
    
    This model combines BRISQUE, Aesthetic Predictor, and CLIP features
    to predict human-like image quality scores (1-10 scale).
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Default config
        self.config = config or {
            "clip_embed_dim": 512,
            "hidden_dim": 256,
            "dropout": 0.3,
            "model_type": "deep",
            "output_range": [1.0, 10.0],
            "clip_model": "ViT-B-32",
            "device": "auto"
        }
        
        # Initialize the fusion model
        self.fusion_model = ImageQualityFusionModel(
            clip_embed_dim=self.config["clip_embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            dropout_rate=self.config["dropout"]
        )
        
        # Initialize feature extractor (will be lazy-loaded)
        self._extractor = None
        
        # Model metadata
        self.model_info = {
            "name": "Image Quality Fusion",
            "version": "1.0.0",
            "description": "Multi-modal image quality assessment combining BRISQUE, Aesthetic, and CLIP features",
            "performance": {
                "correlation": 0.52,
                "r2_score": 0.25,
                "mae": 1.43
            },
            "training_data": "SPAQ dataset (11,125 smartphone images)"
        }
    
    @property
    def extractor(self):
        """Lazy-load the feature extractor"""
        if self._extractor is None:
            self._extractor = ImageQualityExtractor()
        return self._extractor
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion model"""
        return self.fusion_model(features)
    
    def predict_quality(self, image: Union[str, Image.Image, np.ndarray]) -> float:
        """
        Predict image quality score from an image.
        
        Args:
            image: Path to image, PIL Image, or numpy array
            
        Returns:
            Quality score between 1.0 and 10.0
        """
        # Extract features
        features = self.extractor.extract_features_single_image(image)
        
        # Convert to tensors
        features_tensor = {
            'brisque': torch.tensor([features['brisque_normalized']], dtype=torch.float32),
            'laion': torch.tensor([features['aesthetic_normalized']], dtype=torch.float32),
            'clip': torch.tensor(features['clip_embedding'], dtype=torch.float32).unsqueeze(0)
        }
        
        # Move to device
        device = next(self.parameters()).device
        features_tensor = {k: v.to(device) for k, v in features_tensor.items()}
        
        # Predict
        with torch.no_grad():
            prediction = self.forward(features_tensor)
            return prediction.item()
    
    def predict_batch(self, images: list) -> list:
        """
        Predict quality scores for a batch of images.
        
        Args:
            images: List of image paths, PIL Images, or numpy arrays
            
        Returns:
            List of quality scores between 1.0 and 10.0
        """
        scores = []
        for image in images:
            score = self.predict_quality(image)
            scores.append(score)
        return scores
    
    @classmethod
    def from_pretrained_fusion(cls, model_path: str, **kwargs):
        """
        Load from a trained fusion model checkpoint.
        
        Args:
            model_path: Path to the trained model (.pth file)
            **kwargs: Additional arguments
            
        Returns:
            ImageQualityFusionHF instance
        """
        # Load the trained model
        model, metadata = ImageQualityFusionModel.load_model(model_path)
        
        # Create HF wrapper
        config = {
            "clip_embed_dim": 512,
            "hidden_dim": 256,
            "dropout": 0.3,
            "model_type": "deep",
            "output_range": [1.0, 10.0],
            "clip_model": "ViT-B-32"
        }
        
        # Update config with metadata if available
        if metadata:
            config.update(metadata.get("config", {}))
        
        hf_model = cls(config=config)
        hf_model.fusion_model.load_state_dict(model.state_dict())
        
        # Add training metadata
        if metadata:
            hf_model.model_info.update({
                "training_metadata": metadata,
                "performance": metadata.get("metrics", hf_model.model_info["performance"])
            })
        
        return hf_model
    
    def get_model_info(self) -> Dict:
        """Get model information and metadata"""
        return self.model_info
    
    def _save_pretrained(self, save_directory: str) -> None:
        """Save additional files alongside the model"""
        # Save model info
        info_path = f"{save_directory}/model_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)


# Convenience function for easy loading
def load_fusion_model_for_hf(model_path: str) -> ImageQualityFusionHF:
    """
    Load a trained fusion model and wrap it for Hugging Face Hub.
    
    Args:
        model_path: Path to the trained model (.pth file)
        
    Returns:
        ImageQualityFusionHF ready for hub upload
    """
    return ImageQualityFusionHF.from_pretrained_fusion(model_path)