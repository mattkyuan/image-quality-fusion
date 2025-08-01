# src/image_quality_fusion/models/brisque_opencv.py
"""
OpenCV BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
Using the official OpenCV implementation for better accuracy and compatibility.

Based on the original paper:
"No-Reference Image Quality Assessment in the Spatial Domain" by Mittal et al.
"""

import cv2
import numpy as np
from PIL import Image
import warnings
from typing import Tuple
from pathlib import Path
import os
import requests
from tqdm import tqdm


class OpenCVBRISQUEModel:
    """
    OpenCV-based BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    
    This implementation uses OpenCV's built-in cv2.quality.QualityBRISQUE
    which provides the official implementation with pre-trained models.
    
    Lower scores indicate better image quality (typically 0-100 range).
    """
    
    # URLs for the official BRISQUE model files from OpenCV
    MODEL_FILES = {
        'range_file': 'https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml',
        'model_file': 'https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml'
    }
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize OpenCV BRISQUE model.
        
        Args:
            cache_dir: Directory to cache model files (default: ~/.cache/opencv_brisque)
        """
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/opencv_brisque')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and setup model files
        self._setup_model_files()
        
        # Initialize BRISQUE quality assessor
        self.brisque_assessor = cv2.quality.QualityBRISQUE_create(
            str(self.model_path),
            str(self.range_path)
        )
        
        print("OpenCV BRISQUE model initialized successfully")
    
    def _setup_model_files(self):
        """Download and setup the required BRISQUE model files."""
        self.model_path = self.cache_dir / "brisque_model_live.yml"
        self.range_path = self.cache_dir / "brisque_range_live.yml"
        
        # Download model file if not exists
        if not self.model_path.exists():
            print("Downloading BRISQUE model file...")
            self._download_file(
                self.MODEL_FILES['model_file'], 
                self.model_path
            )
        
        # Download range file if not exists
        if not self.range_path.exists():
            print("Downloading BRISQUE range file...")
            self._download_file(
                self.MODEL_FILES['range_file'], 
                self.range_path
            )
    
    def _download_file(self, url: str, save_path: Path):
        """Download a file with progress bar."""
        try:
            print(f"Downloading from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size if available
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
            
            print(f"File downloaded: {save_path}")
            
        except Exception as e:
            if save_path.exists():
                save_path.unlink()
            raise RuntimeError(f"Failed to download file: {e}")
    
    def calculate_brisque_score(self, image_path: str) -> float:
        """
        Calculate BRISQUE score for an image using OpenCV's implementation.
        
        Args:
            image_path: Path to image file
            
        Returns:
            float: BRISQUE score (lower = better quality)
        """
        try:
            # Load and validate image
            if isinstance(image_path, str):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    # Try with PIL as fallback
                    pil_img = Image.open(image_path).convert('RGB')
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                raise ValueError("image_path must be a string path to an image file")
                
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Calculate BRISQUE score using OpenCV
            score = self.brisque_assessor.compute(image)
            
            # OpenCV returns a tuple with score and quality map
            if isinstance(score, tuple):
                return float(score[0])
            else:
                return float(score)
                
        except Exception as e:
            warnings.warn(f"Error calculating BRISQUE score for {image_path}: {e}")
            return 50.0  # Return neutral score on error
    
    def calculate_brisque_score_from_array(self, image_array: np.ndarray) -> float:
        """
        Calculate BRISQUE score from image array.
        
        Args:
            image_array: Image as numpy array (BGR format for OpenCV)
            
        Returns:
            float: BRISQUE score (lower = better quality)
        """
        try:
            if image_array is None:
                raise ValueError("Image array is None")
            
            # Ensure the array is in the right format
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Color image
                pass
            elif len(image_array.shape) == 2:
                # Grayscale - convert to 3-channel
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                raise ValueError(f"Unsupported image shape: {image_array.shape}")
            
            # Ensure uint8 format
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    # Assume normalized to [0,1]
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Calculate BRISQUE score
            score = self.brisque_assessor.compute(image_array)
            
            # OpenCV returns a tuple with score and quality map
            if isinstance(score, tuple):
                return float(score[0])
            else:
                return float(score)
                
        except Exception as e:
            warnings.warn(f"Error calculating BRISQUE score from array: {e}")
            return 50.0
    
    def normalize_score(self, score: float, target_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """
        Normalize BRISQUE score to target range.
        
        Args:
            score: Raw BRISQUE score
            target_range: Target range tuple (min, max)
            
        Returns:
            float: Normalized score (0=best, 1=worst in default range)
        """
        # OpenCV BRISQUE scores typically range from 0 (perfect) to 100 (very poor)
        source_min, source_max = 0.0, 100.0
        target_min, target_max = target_range
        
        # Clamp to source range
        score = max(source_min, min(source_max, score))
        
        # Normalize to 0-1
        normalized = score / source_max
        
        # Scale to target range
        scaled = target_min + normalized * (target_max - target_min)
        
        return float(np.clip(scaled, target_min, target_max))
    
    def get_quality_description(self, score: float) -> str:
        """
        Get human-readable quality description.
        
        Args:
            score: Raw BRISQUE score
            
        Returns:
            str: Quality description
        """
        # BRISQUE score ranges (based on empirical observations)
        if score <= 20:
            return "Excellent"
        elif score <= 35:
            return "Good"
        elif score <= 50:
            return "Fair"
        elif score <= 65:
            return "Poor"
        else:
            return "Very Poor"
    
    def batch_calculate_brisque_scores(self, image_paths: list) -> list:
        """
        Calculate BRISQUE scores for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of BRISQUE scores
        """
        scores = []
        print(f"Processing {len(image_paths)} images with OpenCV BRISQUE...")
        
        for image_path in tqdm(image_paths, desc="Computing BRISQUE scores"):
            score = self.calculate_brisque_score(image_path)
            scores.append(score)
        
        return scores
    
    def get_model_info(self) -> dict:
        """Get information about the BRISQUE model."""
        return {
            'implementation': 'opencv',
            'model_type': 'official-opencv-brisque',
            'model_files': {
                'model_path': str(self.model_path),
                'range_path': str(self.range_path)
            },
            'cache_dir': str(self.cache_dir),
            'opencv_version': cv2.__version__,
            'score_range': '0-100 (lower is better)'
        }


def get_opencv_brisque_model(cache_dir: str = None) -> OpenCVBRISQUEModel:
    """
    Convenience function to get OpenCV BRISQUE Model.
    
    Args:
        cache_dir: Cache directory for model files
        
    Returns:
        OpenCVBRISQUEModel: Configured BRISQUE model
    """
    return OpenCVBRISQUEModel(cache_dir=cache_dir)


# Backward compatibility aliases
BRISQUEModel = OpenCVBRISQUEModel