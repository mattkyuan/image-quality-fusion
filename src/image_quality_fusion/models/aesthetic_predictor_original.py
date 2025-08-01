# src/image_quality_fusion/models/aesthetic_predictor_original.py
"""
Aesthetic Predictor based on christophschuhmann/improved-aesthetic-predictor
Original implementation adapted for our pipeline.

Based on: https://github.com/christophschuhmann/improved-aesthetic-predictor
Using their exact model architecture and inference approach.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import open_clip
from PIL import Image
import numpy as np
from pathlib import Path
import os
import requests
from typing import Optional, List
from tqdm import tqdm
import warnings


def normalized(a, axis=-1, order=2):
    """Normalize array along specified axis."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class AestheticMLP(pl.LightningModule):
    """
    MLP for aesthetic prediction - exact architecture from original repo.
    """
    
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        
        # Exact architecture from improved-aesthetic-predictor
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class OriginalAestheticPredictor:
    """
    Aesthetic Predictor using the original implementation approach.
    This closely follows the simple_inference.py from the original repo.
    """
    
    # Model configurations - using original model URLs from both repos
    MODEL_CONFIGS = {
        'ViT-L-14': {
            'clip_model': 'ViT-L-14',
            'pretrained': 'openai',
            'embed_dim': 768,
            'model_url': 'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth',
            'model_type': 'christophschuhmann_mlp'
        },
        'ViT-B-32': {
            'clip_model': 'ViT-B-32',
            'pretrained': 'openai', 
            'embed_dim': 512,
            'model_url': 'https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth',
            'model_type': 'laion_linear'
        }
    }
    
    def __init__(
        self,
        clip_model_name: str = 'ViT-L-14',
        device: str = 'auto',
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Original Aesthetic Predictor following the original approach.
        
        Args:
            clip_model_name: CLIP model name (currently supports 'ViT-L-14')
            device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
            cache_dir: Directory to cache models
        """
        if clip_model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported CLIP model: {clip_model_name}. Supported: {list(self.MODEL_CONFIGS.keys())}")
        
        self.clip_model_name = clip_model_name
        self.config = self.MODEL_CONFIGS[clip_model_name]
        self.device = self._setup_device(device)
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/improved_aesthetic_predictor')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models using original approach
        self._load_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _load_models(self):
        """Load CLIP and aesthetic models using original approach"""
        print(f"Loading CLIP model: {self.clip_model_name}")
        
        # Load CLIP model using open-clip (similar to original but using open-clip)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config['clip_model'],
            pretrained=self.config['pretrained'],
            device=self.device
        )
        self.clip_model.eval()
        
        # Load aesthetic model (MLP or Linear depending on type)
        print(f"Loading aesthetic model ({self.config['model_type']})...")
        if self.config['model_type'] == 'christophschuhmann_mlp':
            self.aesthetic_model = AestheticMLP(self.config['embed_dim'])
        else:  # laion_linear
            # Simple linear model like LAION-AI aesthetic-predictor
            self.aesthetic_model = nn.Linear(self.config['embed_dim'], 1)
        
        # Download and load pre-trained weights
        model_path = self.cache_dir / f"aesthetic_model_{self.clip_model_name.replace('-', '_')}.pth"
        
        if not model_path.exists():
            print("Downloading aesthetic model weights...")
            self._download_model(self.config['model_url'], model_path)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.aesthetic_model.load_state_dict(state_dict)
        self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()
        
        print("Models loaded successfully")
    
    def _download_model(self, url: str, save_path: Path):
        """Download model file with progress bar"""
        try:
            print(f"Downloading from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"Model downloaded: {save_path}")
            
        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")
    
    def calculate_aesthetic_score(self, image_path: str) -> float:
        """
        Calculate aesthetic score using original inference approach.
        
        Args:
            image_path: Path to image file
            
        Returns:
            float: Aesthetic score
        """
        try:
            # Load and preprocess image (following original approach)
            pil_image = Image.open(image_path).convert('RGB')
            image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Extract CLIP features (following original)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                
                # Normalize features (using original normalization function)
                im_emb_arr = normalized(image_features.cpu().detach().numpy())
                
                # Predict aesthetic score
                prediction = self.aesthetic_model(
                    torch.from_numpy(im_emb_arr).to(self.device).float()
                )
                
            return float(prediction.cpu().numpy()[0, 0])
            
        except Exception as e:
            warnings.warn(f"Error calculating aesthetic score for {image_path}: {e}")
            return 5.0  # Return neutral score on error
    
    def calculate_aesthetic_score_from_clip_features(self, clip_features: torch.Tensor) -> float:
        """
        Calculate aesthetic score from pre-computed CLIP features.
        
        Args:
            clip_features: CLIP image features
            
        Returns:
            float: Aesthetic score
        """
        try:
            if clip_features.dim() == 1:
                clip_features = clip_features.unsqueeze(0)
            
            # Normalize features using original method
            features_np = clip_features.cpu().detach().numpy()
            im_emb_arr = normalized(features_np)
            
            with torch.no_grad():
                prediction = self.aesthetic_model(
                    torch.from_numpy(im_emb_arr).to(self.device).float()
                )
            
            return float(prediction.cpu().numpy()[0, 0])
            
        except Exception as e:
            warnings.warn(f"Error calculating aesthetic score from features: {e}")
            return 5.0
    
    def batch_calculate_aesthetic_scores(self, image_paths: List[str]) -> List[float]:
        """Calculate aesthetic scores for multiple images"""
        scores = []
        print(f"Processing {len(image_paths)} images...")
        
        for image_path in tqdm(image_paths, desc="Computing aesthetic scores"):
            score = self.calculate_aesthetic_score(image_path)
            scores.append(score)
        
        return scores
    
    def normalize_score(self, score: float, target_range: tuple = (0.0, 1.0)) -> float:
        """
        Normalize aesthetic score to target range.
        
        Based on the original model's output range (typically 1-10).
        """
        # Original model typically outputs scores in 1-10 range
        source_min, source_max = 1.0, 10.0
        target_min, target_max = target_range
        
        # Clamp to source range
        score = max(source_min, min(source_max, score))
        
        # Normalize to 0-1
        normalized = (score - source_min) / (source_max - source_min)
        
        # Scale to target range
        scaled = target_min + normalized * (target_max - target_min)
        
        return float(np.clip(scaled, target_min, target_max))
    
    def get_aesthetic_description(self, score: float) -> str:
        """Get human-readable aesthetic description"""
        if score >= 8.0:
            return "Highly Aesthetic"
        elif score >= 7.0:
            return "Very Pleasing"
        elif score >= 6.0:
            return "Pleasant"
        elif score >= 5.0:
            return "Average"
        elif score >= 4.0:  
            return "Below Average"
        else:
            return "Poor Aesthetics"
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        if self.config['model_type'] == 'christophschuhmann_mlp':
            architecture = f'MLP: {self.config["embed_dim"]}->1024->128->64->16->1'
        else:
            architecture = f'Linear: {self.config["embed_dim"]}->1'
            
        return {
            'clip_model': self.clip_model_name,
            'embed_dim': self.config['embed_dim'],
            'device': self.device,
            'cache_dir': str(self.cache_dir),
            'model_type': f'original-{self.config["model_type"]}',
            'architecture': architecture
        }


def get_original_aesthetic_predictor(
    clip_model: str = 'ViT-L-14',
    device: str = 'auto',
    cache_dir: Optional[str] = None
) -> OriginalAestheticPredictor:
    """
    Convenience function to get Original Aesthetic Predictor
    
    Args:
        clip_model: CLIP model name
        device: Device to run on
        cache_dir: Cache directory for models
        
    Returns:
        OriginalAestheticPredictor: Configured aesthetic predictor
    """
    return OriginalAestheticPredictor(
        clip_model_name=clip_model,
        device=device,
        cache_dir=cache_dir
    )


# Backward compatibility alias
ImprovedAestheticPredictor = OriginalAestheticPredictor
LAIONAestheticsModel = OriginalAestheticPredictor