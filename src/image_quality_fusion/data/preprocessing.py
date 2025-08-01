# src/image_quality_fusion/data/preprocessing.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from ..models.brisque_opencv import OpenCVBRISQUEModel
from ..models.aesthetic_predictor_original import OriginalAestheticPredictor
from ..models.clip_model import CLIPSemanticAnalyzer

# Global BRISQUE model for multiprocessing
_global_brisque_model = None

def _init_brisque_worker():
    """Initialize BRISQUE model for worker process"""
    global _global_brisque_model
    if _global_brisque_model is None:
        _global_brisque_model = OpenCVBRISQUEModel()

def _compute_brisque_score(path):
    """Compute BRISQUE score for a single image (worker function)"""
    try:
        return _global_brisque_model.calculate_brisque_score(str(path))
    except Exception as e:
        print(f"BRISQUE failed for {path}: {e}")
        return 50.0  # Default fallback score

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
    
    def extract_features_batch(self, image_paths, output_path=None, batch_size=3000, use_cache=True):
        """Extract features for multiple images with M1 optimization and improved caching"""
        import time
        
        results = []
        total_images = len(image_paths)
        
        print(f"Processing {total_images} images with M1 optimizations...")
        print(f"Batch size: {batch_size} (optimized for M1 Pro GPU)")
        
        # Check for existing cache
        if use_cache and output_path:
            cache_features_path = Path(output_path)
            cache_embeddings_path = Path(output_path.replace('.csv', '_embeddings.npy'))
            
            if cache_features_path.exists() and cache_embeddings_path.exists():
                print(f"Loading cached features from {cache_features_path}")
                try:
                    cached_df = pd.read_csv(cache_features_path)
                    cached_embeddings = np.load(cache_embeddings_path)
                    
                    # Verify cache integrity
                    if len(cached_df) == len(cached_embeddings) == len(image_paths):
                        # Restore clip_embedding column for consistency
                        cached_df['clip_embedding'] = [emb for emb in cached_embeddings]
                        print(f"Successfully loaded {len(cached_df)} cached features")
                        return cached_df
                    else:
                        print("Cache size mismatch, recomputing features...")
                except Exception as e:
                    print(f"Cache loading failed: {e}, recomputing features...")
        
        # Track processing time
        start_time = time.time()
        processed_count = 0
        
        # Process in optimized batches for M1
        for start_idx in range(0, total_images, batch_size):
            end_idx = min(start_idx + batch_size, total_images)
            batch_paths = image_paths[start_idx:end_idx]
            
            batch_start = time.time()
            print(f"Processing batch {start_idx//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            # Process batch with optimized method
            batch_results = self._extract_features_batch_optimized(batch_paths)
            results.extend(batch_results)
            processed_count += len(batch_results)
            
            batch_time = time.time() - batch_start
            avg_time_per_image = batch_time / len(batch_results) if batch_results else 0
            print(f"Batch completed in {batch_time:.1f}s ({avg_time_per_image:.2f}s per image)")
            
            # Memory management for large datasets
            if len(results) % (batch_size * 2) == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        print(f"Feature extraction completed in {total_time:.1f}s ({total_time/processed_count:.2f}s per image)")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        if output_path:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save CLIP embeddings separately (they're large)
            embeddings = np.stack(df['clip_embedding'].values)
            embeddings_path = output_path.replace('.csv', '_embeddings.npy')
            
            # Also save with standard naming for compatibility
            standard_embeddings_path = output_path.replace('.csv', '.npy').replace('/features.', '/embeddings.')
            if 'features.csv' in output_path:
                standard_embeddings_path = output_path.replace('features.csv', 'embeddings.npy')
            
            np.save(embeddings_path, embeddings)
            np.save(standard_embeddings_path, embeddings)  # Compatibility naming
            
            # Save other features as CSV
            df_without_embeddings = df.drop('clip_embedding', axis=1)
            df_without_embeddings.to_csv(output_path, index=False)
            print(f"✅ Features saved to {output_path}")
            print(f"✅ Embeddings saved to {embeddings_path}")
            print(f"💾 Cache ready for fast loading in future runs")
            
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
        
        # Batch process BRISQUE scores in parallel (CPU-bound)
        brisque_scores = self._batch_brisque_parallel(valid_paths)
        
        # Batch process aesthetic scores using CLIP features (GPU/MPS accelerated)
        aesthetic_scores = self.laion_model.batch_calculate_aesthetic_scores_from_clip(clip_embeddings)
        
        # Process other features
        for i, (image_path, image) in enumerate(zip(valid_paths, images)):
            try:
                # Use pre-computed scores
                brisque_score = brisque_scores[i]
                aesthetic_score = aesthetic_scores[i]
                
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
    
    def _batch_brisque_parallel(self, image_paths, num_workers=None):
        """Parallel BRISQUE processing using multiprocessing with M1 optimization"""
        if num_workers is None:
            # For M1 MacBook Pro: use 6-8 workers (leave 2 cores for main process)
            num_workers = min(8, max(1, cpu_count() - 2), len(image_paths))
        
        print(f"Processing BRISQUE scores with {num_workers} workers on M1...")
        
        # For small batches, use sequential processing to avoid overhead
        if len(image_paths) < num_workers * 2:
            print("Small batch detected, using sequential BRISQUE processing")
            brisque_model = OpenCVBRISQUEModel()
            return [brisque_model.calculate_brisque_score(str(path)) for path in image_paths]
        
        # Use multiprocessing pool with worker initialization
        try:
            with Pool(processes=num_workers, initializer=_init_brisque_worker) as pool:
                brisque_scores = pool.map(_compute_brisque_score, image_paths)
            return brisque_scores
        except Exception as e:
            print(f"Parallel BRISQUE processing failed: {e}, falling back to sequential")
            # Fallback to sequential processing
            brisque_model = OpenCVBRISQUEModel()
            return [brisque_model.calculate_brisque_score(str(path)) for path in image_paths]
