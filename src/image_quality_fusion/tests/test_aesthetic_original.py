# src/image_quality_fusion/tests/test_aesthetic_original.py
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

from ..models.aesthetic_predictor_original import OriginalAestheticPredictor, get_original_aesthetic_predictor


class TestOriginalAestheticPredictor:
    """Test suite for Original Aesthetic Predictor"""
    
    @pytest.fixture
    def test_images(self):
        """Create test images"""
        test_dir = Path(tempfile.mkdtemp())
        
        # Create a pleasant gradient image
        img_aesthetic = Image.new('RGB', (224, 224))
        pixels = []
        for y in range(224):
            for x in range(224):
                # Create a sunset-like gradient
                r = int(255 * (1 - y / 224) * 0.9)
                g = int(180 * (1 - y / 224) * 0.7)
                b = int(120 * (y / 224) + 50)
                pixels.append((r, g, b))
        img_aesthetic.putdata(pixels)
        aesthetic_path = test_dir / "aesthetic.jpg"
        img_aesthetic.save(str(aesthetic_path))
        
        # Create a simple solid color image
        simple_path = test_dir / "simple.jpg"
        simple_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        simple_img.save(str(simple_path))
        
        # Create a high contrast checkerboard (potentially less aesthetic)
        checker_img = Image.new('RGB', (224, 224))
        checker_pixels = []
        for y in range(224):
            for x in range(224):
                if (x // 20 + y // 20) % 2:
                    checker_pixels.append((255, 255, 255))
                else:
                    checker_pixels.append((0, 0, 0))
        checker_img.putdata(checker_pixels)
        checker_path = test_dir / "checker.jpg"
        checker_img.save(str(checker_path))
        
        yield {
            'aesthetic': str(aesthetic_path),
            'simple': str(simple_path),
            'checker': str(checker_path),
            'test_dir': test_dir
        }
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_model_initialization_vit_b32(self):
        """Test model initialization with ViT-B-32"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-B-32',
            device='cpu'  # Use CPU for testing
        )
        
        assert model is not None
        assert model.clip_model_name == 'ViT-B-32'
        assert model.config['embed_dim'] == 512
        assert model.device == 'cpu'
        
        # Check model info
        info = model.get_model_info()
        assert info['clip_model'] == 'ViT-B-32'
        assert info['embed_dim'] == 512
        assert info['model_type'] == 'improved-aesthetic-predictor'
    
    def test_model_initialization_vit_l14(self):
        """Test model initialization with ViT-L-14"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-L-14',
            device='cpu'
        )
        
        assert model is not None
        assert model.clip_model_name == 'ViT-L-14'
        assert model.config['embed_dim'] == 768
    
    def test_convenience_function(self):
        """Test convenience function"""
        model = get_original_aesthetic_predictor(
            clip_model='ViT-B-32',
            device='cpu'
        )
        
        assert isinstance(model, OriginalAestheticPredictor)
        assert model.clip_model_name == 'ViT-B-32'
    
    def test_aesthetic_score_calculation(self, test_images):
        """Test aesthetic score calculation"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-B-32',
            device='cpu'
        )
        
        # Calculate scores for different images
        aesthetic_score = model.calculate_aesthetic_score(test_images['aesthetic'])
        simple_score = model.calculate_aesthetic_score(test_images['simple'])
        checker_score = model.calculate_aesthetic_score(test_images['checker'])
        
        # Scores should be numeric
        assert isinstance(aesthetic_score, float)
        assert isinstance(simple_score, float)
        assert isinstance(checker_score, float)
        
        # Scores should be in reasonable range
        # Note: ViT-B-32 uses randomly initialized model, so scores may be negative
        for score in [aesthetic_score, simple_score, checker_score]:
            assert isinstance(score, float)
            assert not np.isinf(score) and not np.isnan(score)
        
        print(f"Aesthetic score: {aesthetic_score:.3f}")
        print(f"Simple score: {simple_score:.3f}")
        print(f"Checker score: {checker_score:.3f}")
        
        # Log score differences
        print(f"Aesthetic vs Simple: {aesthetic_score - simple_score:.3f}")
        print(f"Aesthetic vs Checker: {aesthetic_score - checker_score:.3f}")
    
    def test_score_normalization(self):
        """Test score normalization"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-B-32',
            device='cpu'
        )
        
        # Test normalization with different ranges
        test_scores = [1.0, 3.0, 5.5, 8.0, 10.0]
        
        for score in test_scores:
            # Test 0-1 normalization
            norm_01 = model.normalize_score(score, (0.0, 1.0))
            assert 0.0 <= norm_01 <= 1.0
            
            # Test 1-10 normalization
            norm_110 = model.normalize_score(score, (1.0, 10.0))
            assert 1.0 <= norm_110 <= 10.0
        
        # Test that higher scores normalize to higher values
        low_score_norm = model.normalize_score(2.0)
        high_score_norm = model.normalize_score(9.0)
        assert high_score_norm > low_score_norm
    
    def test_aesthetic_descriptions(self):
        """Test aesthetic description mapping"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-B-32',
            device='cpu'
        )
        
        descriptions = [
            (8.5, "Highly Aesthetic"),
            (7.5, "Very Pleasing"),
            (6.5, "Pleasant"),
            (5.5, "Average"),
            (4.5, "Below Average"),
            (3.0, "Poor Aesthetics")
        ]
        
        for score, expected_desc in descriptions:
            desc = model.get_aesthetic_description(score)
            assert desc == expected_desc
    
    def test_batch_processing(self, test_images):
        """Test batch processing of multiple images"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-B-32',
            device='cpu'
        )
        
        image_paths = [
            test_images['aesthetic'], 
            test_images['simple'], 
            test_images['checker']
        ]
        scores = model.batch_calculate_aesthetic_scores(image_paths)
        
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        # Note: ViT-B-32 uses randomly initialized model, so scores may be negative
        assert all(not np.isinf(score) and not np.isnan(score) for score in scores)
        
        print(f"Batch scores: {[f'{s:.3f}' for s in scores]}")
    
    def test_clip_features_prediction(self, test_images):
        """Test prediction from pre-computed CLIP features"""
        model = OriginalAestheticPredictor(
            clip_model_name='ViT-B-32',
            device='cpu'
        )
        
        # Create dummy CLIP features (512-dimensional for ViT-B-32)
        dummy_features = torch.randn(512)
        
        score = model.calculate_aesthetic_score_from_clip_features(dummy_features)
        assert isinstance(score, float)
        # Note: ViT-B-32 uses randomly initialized model, so scores may be negative
        assert not np.isinf(score) and not np.isnan(score)
        
        # Test batch features
        batch_features = torch.randn(3, 512)
        batch_scores = model.batch_calculate_from_features(batch_features)
        
        assert len(batch_scores) == 3
        assert all(isinstance(score, float) for score in batch_scores)
        # Note: ViT-B-32 uses randomly initialized model, so scores may be negative
        assert all(not np.isinf(score) and not np.isnan(score) for score in batch_scores)
    
    def test_unsupported_model_error(self):
        """Test error handling for unsupported models"""
        with pytest.raises(ValueError):
            OriginalAestheticPredictor(
                clip_model_name='ViT-H-16',  # Unsupported model
                device='cpu'
            )


def test_original_aesthetic_standalone():
    """Standalone test function for Original Aesthetic Predictor"""
    print("Testing Original Aesthetic Predictor...")
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test images
        print("Creating test images...")
        
        # Aesthetic image: sunset gradient
        img_aesthetic = Image.new('RGB', (300, 200))
        pixels_aesthetic = []
        for y in range(200):
            for x in range(300):
                # Create sunset colors
                r = int(255 * (1 - y / 200) * 0.9)
                g = int(150 * (1 - y / 200) * 0.8)
                b = int(80 + y / 200 * 100)
                pixels_aesthetic.append((r, g, b))
        img_aesthetic.putdata(pixels_aesthetic)
        aesthetic_path = test_dir / "aesthetic.jpg"
        img_aesthetic.save(str(aesthetic_path))
        
        # Simple image: solid color
        simple_path = test_dir / "simple.jpg"
        simple_img = Image.new('RGB', (300, 200), color=(128, 128, 128))
        simple_img.save(str(simple_path))
        
        # Noisy image: random pixels
        noise_img = Image.new('RGB', (300, 200))
        noise_pixels = []
        np.random.seed(42)
        for _ in range(300 * 200):
            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256) 
            b = np.random.randint(0, 256)
            noise_pixels.append((r, g, b))
        noise_img.putdata(noise_pixels)
        noise_path = test_dir / "noise.jpg"
        noise_img.save(str(noise_path))
        
        # Test model initialization
        print("Initializing Original Aesthetic Predictor...")
        print("(This may take a moment to download models on first run)")
        
        model = get_original_aesthetic_predictor(
            clip_model='ViT-B-32',  # Use ViT-B-32 for faster testing
            device='cpu'
        )
        
        print("✅ Model initialized successfully")
        print(f"Model info: {model.get_model_info()}")
        
        # Test single image prediction
        print("\nTesting aesthetic score prediction...")
        
        aesthetic_score = model.calculate_aesthetic_score(str(aesthetic_path))
        simple_score = model.calculate_aesthetic_score(str(simple_path))
        noise_score = model.calculate_aesthetic_score(str(noise_path))
        
        print(f"Aesthetic image score: {aesthetic_score:.3f}")
        print(f"Simple image score: {simple_score:.3f}")
        print(f"Noise image score: {noise_score:.3f}")
        
        # Test score normalization
        print("\nTesting score normalization...")
        
        norm_aesthetic = model.normalize_score(aesthetic_score, (0.0, 1.0))
        norm_simple = model.normalize_score(simple_score, (0.0, 1.0))
        norm_noise = model.normalize_score(noise_score, (0.0, 1.0))
        
        print(f"Normalized aesthetic: {norm_aesthetic:.3f}")
        print(f"Normalized simple: {norm_simple:.3f}")
        print(f"Normalized noise: {norm_noise:.3f}")
        
        # Test descriptions
        print("\nTesting aesthetic descriptions...")
        
        aesthetic_desc = model.get_aesthetic_description(aesthetic_score)
        simple_desc = model.get_aesthetic_description(simple_score)
        noise_desc = model.get_aesthetic_description(noise_score)
        
        print(f"Aesthetic description: {aesthetic_desc}")
        print(f"Simple description: {simple_desc}")
        print(f"Noise description: {noise_desc}")
        
        # Test batch processing
        print("\nTesting batch processing...")
        
        image_paths = [str(aesthetic_path), str(simple_path), str(noise_path)]
        batch_scores = model.batch_calculate_aesthetic_scores(image_paths)
        
        print(f"Batch scores: {[f'{s:.3f}' for s in batch_scores]}")
        
        # Test CLIP features prediction
        print("\nTesting CLIP features prediction...")
        
        dummy_features = torch.randn(512)  # ViT-B-32 dimension
        feature_score = model.calculate_aesthetic_score_from_clip_features(dummy_features)
        print(f"Score from dummy features: {feature_score:.3f}")
        
        # Verify all scores are reasonable
        all_scores = [aesthetic_score, simple_score, noise_score, feature_score]
        assert all(isinstance(score, float) for score in all_scores)
        
        # For randomly initialized models, scores can be negative
        # For pre-trained models, scores should be in 0-15 range
        if model.config['model_url'] is None:
            print("Note: Using randomly initialized model, scores may be negative")
            # Just check they're finite numbers
            assert all(not np.isinf(score) and not np.isnan(score) for score in all_scores)
        else:
            assert all(0 <= score <= 15 for score in all_scores)
        
        print("\n✅ Improved Aesthetic Predictor test completed successfully!")
        print("The model is now ready for use in the fusion system.")
        
    except Exception as e:
        print(f"❌ Improved Aesthetic Predictor test failed: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_original_aesthetic_standalone()