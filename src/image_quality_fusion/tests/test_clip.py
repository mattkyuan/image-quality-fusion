# src/image_quality_fusion/tests/test_clip.py
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest decorators for standalone execution
    class MockPytest:
        def fixture(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    pytest = MockPytest()

from ..models.clip_model import CLIPSemanticAnalyzer


class TestCLIPSemanticAnalyzer:
    """Test suite for CLIP Semantic Analyzer"""
    
    @pytest.fixture
    def clip_model(self):
        """Create CLIP model instance"""
        return CLIPSemanticAnalyzer()
    
    @pytest.fixture
    def test_images(self):
        """Create temporary test images"""
        test_dir = Path(tempfile.mkdtemp())
        
        # Sharp, colorful gradient
        img1 = Image.new('RGB', (400, 300))
        pixels1 = []
        for y in range(300):
            for x in range(400):
                r = int(255 * (x / 400))
                g = int(255 * (y / 300))
                b = 128
                pixels1.append((r, g, b))
        img1.putdata(pixels1)
        sharp_path = test_dir / "sharp_colorful.jpg"
        img1.save(str(sharp_path))
        
        # Simple solid color
        simple_path = test_dir / "simple_blue.jpg"
        simple_img = Image.new('RGB', (200, 150), color=(100, 150, 255))
        simple_img.save(str(simple_path))
        
        # High contrast checkerboard
        img3 = Image.new('RGB', (300, 200))
        pixels3 = []
        for y in range(200):
            for x in range(300):
                if (x // 20 + y // 20) % 2:
                    pixels3.append((255, 255, 255))
                else:
                    pixels3.append((0, 0, 0))
        img3.putdata(pixels3)
        checker_path = test_dir / "checkerboard.jpg"
        img3.save(str(checker_path))
        
        yield {
            'sharp_colorful': str(sharp_path),
            'simple_blue': str(simple_path),
            'checkerboard': str(checker_path),
            'test_dir': test_dir
        }
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_clip_initialization(self, clip_model):
        """Test CLIP model initialization"""
        assert clip_model is not None
        assert hasattr(clip_model, 'model')
        assert hasattr(clip_model, 'preprocess')
        assert hasattr(clip_model, 'tokenizer')
        
        # Check embedding dimension
        embed_dim = clip_model.get_embedding_dim()
        assert isinstance(embed_dim, int)
        assert embed_dim > 0
    
    def test_encode_image(self, clip_model, test_images):
        """Test image encoding"""
        for image_name, image_path in test_images.items():
            if image_name == 'test_dir':
                continue
                
            embedding = clip_model.encode_image(image_path)
            
            # Check embedding properties
            assert isinstance(embedding, torch.Tensor)
            assert embedding.dim() == 1  # Should be 1D vector
            assert len(embedding) > 0
            
            # Check embedding is normalized (should have norm close to 1)
            norm = torch.norm(embedding)
            assert abs(norm - 1.0) < 0.1  # Allow small tolerance
    
    def test_encode_text(self, clip_model):
        """Test text encoding"""
        test_texts = [
            "a colorful image",
            "a simple photo",
            "a black and white pattern",
            "a blue image"
        ]
        
        embeddings = clip_model.encode_text(test_texts)
        
        # Check embeddings properties
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 2  # Should be 2D matrix
        assert embeddings.shape[0] == len(test_texts)
        assert embeddings.shape[1] > 0
        
        # Check embeddings are normalized
        norms = torch.norm(embeddings, dim=1)
        for norm in norms:
            assert abs(norm - 1.0) < 0.1
    
    def test_compute_similarity(self, clip_model, test_images):
        """Test similarity computation"""
        descriptions = [
            "a colorful image",
            "a simple photo", 
            "a black and white pattern",
            "a blue image"
        ]
        
        # Test with different images
        for image_name, image_path in test_images.items():
            if image_name == 'test_dir':
                continue
                
            similarities = clip_model.compute_similarity(image_path, descriptions)
            
            # Check similarity properties
            assert isinstance(similarities, torch.Tensor)
            assert similarities.dim() == 1
            assert len(similarities) == len(descriptions)
            
            # Similarities should be in reasonable range [-1, 1]
            assert torch.all(similarities >= -1.0)
            assert torch.all(similarities <= 1.0)
    
    def test_semantic_matching(self, clip_model, test_images):
        """Test that CLIP makes reasonable semantic matches"""
        descriptions = [
            "a colorful gradient",
            "a simple blue image",
            "a black and white checkerboard pattern"
        ]
        
        # Test blue image should match "simple blue image" best
        similarities_blue = clip_model.compute_similarity(
            test_images['simple_blue'], descriptions
        )
        best_match_blue = torch.argmax(similarities_blue)
        # Should match description index 1 ("a simple blue image")
        assert best_match_blue == 1
        
        # Test checkerboard should match "black and white pattern" best
        similarities_checker = clip_model.compute_similarity(
            test_images['checkerboard'], descriptions
        )
        best_match_checker = torch.argmax(similarities_checker)
        # Should match description index 2 ("black and white checkerboard pattern")
        assert best_match_checker == 2
    
    def test_invalid_image_path(self, clip_model):
        """Test handling of invalid image paths"""
        embedding = clip_model.encode_image("nonexistent_image.jpg")
        # Should return zero embedding on error
        assert torch.allclose(embedding, torch.zeros_like(embedding))


def test_clip_standalone():
    """Standalone test function for backwards compatibility"""
    print("Testing CLIP model...")
    
    # Create test images
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Sharp, colorful gradient
        img1 = Image.new('RGB', (400, 300))
        pixels1 = []
        for y in range(300):
            for x in range(400):
                r = int(255 * (x / 400))
                g = int(255 * (y / 300))
                b = 128
                pixels1.append((r, g, b))
        img1.putdata(pixels1)
        sharp_path = test_dir / "sharp_colorful.jpg"
        img1.save(str(sharp_path))
        
        # Simple solid color
        simple_path = test_dir / "simple_blue.jpg"
        simple_img = Image.new('RGB', (200, 150), color=(100, 150, 255))
        simple_img.save(str(simple_path))
        
        # High contrast checkerboard
        img3 = Image.new('RGB', (300, 200))
        pixels3 = []
        for y in range(200):
            for x in range(300):
                if (x // 20 + y // 20) % 2:
                    pixels3.append((255, 255, 255))
                else:
                    pixels3.append((0, 0, 0))
        img3.putdata(pixels3)
        checker_path = test_dir / "checkerboard.jpg"
        img3.save(str(checker_path))
        
        test_files = [str(sharp_path), str(simple_path), str(checker_path)]
        
        # Initialize CLIP
        clip_model = CLIPSemanticAnalyzer()
        print(f"CLIP embedding dimension: {clip_model.get_embedding_dim()}")
        
        # Test each image
        for image_path in test_files:
            filename = Path(image_path).name
            print(f"\nProcessing: {filename}")
            
            # Get embedding
            embedding = clip_model.encode_image(image_path)
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {torch.norm(embedding):.3f}")
            
            # Test text similarity
            descriptions = [
                "a colorful image",
                "a simple photo", 
                "a black and white pattern",
                "a blue image"
            ]
            
            similarities = clip_model.compute_similarity(image_path, descriptions)
            print(f"  Similarities: {similarities.numpy()}")
            
            # Find best description
            best_idx = torch.argmax(similarities)
            print(f"  Best description: '{descriptions[best_idx]}' (score: {similarities[best_idx]:.3f})")
        
        print("\n✅ CLIP test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing CLIP: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_clip_standalone()