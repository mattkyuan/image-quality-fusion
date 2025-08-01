# src/image_quality_fusion/data/preprocessing.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from ..models.brisque_opencv import OpenCVBRISQUEModel
from ..models.aesthetic_predictor_original import OriginalAestheticPredictor
from ..models.clip_model import CLIPSemanticAnalyzer

class ImageQualityExtractor:
    def __init__(self, clip_model_name: str = 'ViT-B-32'):
        """
        Initialize image quality extractor
        
        Args:
            clip_model_name: CLIP model to use ('ViT-B-32' or 'ViT-L-14')
        """
        self.brisque_model = OpenCVBRISQUEModel()
        self.laion_model = OriginalAestheticPredictor(clip_model_name=clip_model_name)
        self.clip_model = CLIPSemanticAnalyzer(model_name=clip_model_name)
        
    def extract_features_single_image(self, image_path):
        """Extract all features for a single image"""
        try:
            # Validate image
            with Image.open(image_path) as img:
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                    
            # Extract features
            brisque_score = self.brisque_model.calculate_brisque_score(image_path)
            aesthetic_score = self.laion_model.calculate_aesthetic_score(image_path)
            clip_embedding = self.clip_model.encode_image(image_path)  # Added CLIP
            
            # Normalize scores
            brisque_norm = self.brisque_model.normalize_score(brisque_score)
            aesthetic_norm = self.laion_model.normalize_score(aesthetic_score)
            
            return {
                'image_path': str(image_path),
                'brisque_raw': brisque_score,
                'aesthetic_raw': aesthetic_score,
                'brisque_normalized': brisque_norm,
                'aesthetic_normalized': aesthetic_norm,
                'clip_embedding': clip_embedding.numpy(),  # Convert to numpy for storage
                'technical_quality': brisque_norm,
                'aesthetic_quality': aesthetic_norm,
                'embedding_dim': len(clip_embedding)
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_features_batch(self, image_paths, output_path=None, batch_size=500):
        """Extract features for multiple images with M1 optimization"""
        results = []
        total_images = len(image_paths)
        
        print(f"Processing {total_images} images with M1 optimizations...")
        print(f"Batch size: {batch_size} (optimized for 32GB RAM)")
        
        # Process in optimized batches for M1
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            batch_paths = image_paths[start_idx:end_idx]
            
            print(f"Processing batch {start_idx//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            # Process batch with optimized method
            batch_results = self._extract_features_batch_optimized(batch_paths)
            results.extend(batch_results)
            
            # Memory management for large datasets
            if len(results) % (batch_size * 2) == 0:
                torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if output_path:
            # Save CLIP embeddings separately (they're large)
            embeddings = np.stack(df['clip_embedding'].values)
            np.save(output_path.replace('.csv', '_embeddings.npy'), embeddings)
            
            # Save other features as CSV
            df_without_embeddings = df.drop('clip_embedding', axis=1)
            df_without_embeddings.to_csv(output_path, index=False)
            print(f"Features saved to {output_path}")
            print(f"Embeddings saved to {output_path.replace('.csv', '_embeddings.npy')}")
            
        return df
    
    def _extract_features_batch_optimized(self, image_paths):
        """Optimized batch processing for M1 MacBook Pro"""
        batch_results = []
        
        # Pre-load images in memory (with 32GB RAM, we can afford this)
        images = []
        valid_paths = []
        
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    if img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                    images.append(img.copy())
                    valid_paths.append(image_path)
            except Exception as e:
                print(f"Failed to load {image_path}: {e}")
                continue
        
        if not images:
            return batch_results
        
        # Batch process CLIP embeddings (GPU/MPS accelerated)
        clip_embeddings = self._batch_clip_embeddings(images)
        
        # Process other features (can be parallelized on CPU)
        for i, (image_path, image) in enumerate(zip(valid_paths, images)):
            try:
                # BRISQUE (CPU-bound)
                brisque_score = self.brisque_model.calculate_brisque_score(image_path)
                
                # Aesthetic score (GPU/MPS accelerated, but single image)
                aesthetic_score = self.laion_model.calculate_aesthetic_score(image_path)
                
                # Use pre-computed CLIP embedding
                clip_embedding = clip_embeddings[i]
                
                # Normalize scores
                brisque_norm = self.brisque_model.normalize_score(brisque_score)
                aesthetic_norm = self.laion_model.normalize_score(aesthetic_score)
                
                batch_results.append({
                    'image_path': str(image_path),
                    'brisque_raw': brisque_score,
                    'aesthetic_raw': aesthetic_score,
                    'brisque_normalized': brisque_norm,
                    'aesthetic_normalized': aesthetic_norm,
                    'clip_embedding': clip_embedding.numpy(),
                    'technical_quality': brisque_norm,
                    'aesthetic_quality': aesthetic_norm,
                    'embedding_dim': len(clip_embedding)
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return batch_results
    
    def _batch_clip_embeddings(self, images):
        """Batch process CLIP embeddings with MPS acceleration"""
        try:
            # Convert PIL images to tensors
            image_tensors = []
            for img in images:
                tensor = self.clip_model.preprocess(img).unsqueeze(0)
                image_tensors.append(tensor)
            
            # Stack into batch tensor
            batch_tensor = torch.cat(image_tensors, dim=0)
            
            # Move to MPS if available
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            batch_tensor = batch_tensor.to(device)
            
            # Process batch through CLIP
            with torch.no_grad():
                embeddings = self.clip_model.model.encode_image(batch_tensor)
                embeddings = embeddings.cpu()  # Move back to CPU for numpy conversion
            
            # Return list of individual embeddings
            return [embeddings[i] for i in range(len(embeddings))]
            
        except Exception as e:
            print(f"Batch CLIP processing failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            embeddings = []
            for img in images:
                try:
                    embedding = self.clip_model.encode_image(img)
                    embeddings.append(embedding)
                except:
                    # Create zero embedding as fallback
                    embedding_dim = self.clip_model.get_embedding_dim()
                    embeddings.append(torch.zeros(embedding_dim))
            return embeddings
