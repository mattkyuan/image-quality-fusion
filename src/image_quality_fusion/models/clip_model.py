# src/models/clip_model.py
import torch
import torch.nn as nn
import open_clip
from PIL import Image
import numpy as np
from pathlib import Path

class CLIPSemanticAnalyzer:
    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        """
        CLIP model for semantic analysis of images
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and preprocessing"""
        try:
            print(f"Loading CLIP model: {self.model_name}")
            
            # Load model, tokenizer, and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=self.device
            )
            
            # Get tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            print("CLIP model loaded successfully")
            
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
    
    def encode_image(self, image_input):
        """
        Extract CLIP embeddings for an image
        
        Args:
            image_input: Either a PIL Image or path to image file
            
        Returns:
            torch.Tensor: Image embeddings [embed_dim]
        """
        try:
            # Handle both PIL Image and file path inputs
            if isinstance(image_input, (str, Path)):
                # Load and preprocess image from path
                image = Image.open(image_input).convert('RGB')
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            else:
                # Assume PIL Image input
                image_tensor = self.preprocess(image_input).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                
            # Normalize features (standard practice)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.squeeze(0).cpu()  # Remove batch dim, move to CPU
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            # Return zero embedding with correct dimensions
            embed_dim = self.get_embedding_dim()
            return torch.zeros(embed_dim)
    
    def encode_text(self, text_list):
        """
        Encode text descriptions
        
        Args:
            text_list: List of text descriptions
            
        Returns:
            torch.Tensor: Text embeddings [len(text_list), embed_dim]
        """
        try:
            # Tokenize text
            text_tokens = self.tokenizer(text_list).to(self.device)
            
            # Extract features
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu()
            
        except Exception as e:
            print(f"Error encoding text: {e}")
            return torch.zeros(len(text_list), 512)
    
    def compute_similarity(self, image_path, text_descriptions):
        """
        Compute similarity between image and text descriptions
        
        Args:
            image_path: Path to image
            text_descriptions: List of text descriptions
            
        Returns:
            torch.Tensor: Similarity scores [len(text_descriptions)]
        """
        # Get embeddings
        image_features = self.encode_image(image_path).unsqueeze(0)  # [1, embed_dim]
        text_features = self.encode_text(text_descriptions)  # [num_texts, embed_dim]
        
        # Compute similarity (cosine similarity since features are normalized)
        similarity = torch.matmul(image_features, text_features.T).squeeze(0)
        
        return similarity

    def get_embedding_dim(self):
        """Get the dimension of CLIP embeddings"""
        return self.model.visual.output_dim