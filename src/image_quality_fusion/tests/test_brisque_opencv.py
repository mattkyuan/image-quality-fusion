# src/image_quality_fusion/tests/test_brisque_opencv.py
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

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

from ..models.brisque_opencv import OpenCVBRISQUEModel


class TestBRISQUEModel:
    """Test suite for BRISQUE model"""
    
    @pytest.fixture
    def brisque_model(self):
        """Create BRISQUE model instance"""
        return OpenCVBRISQUEModel()
    
    @pytest.fixture
    def test_images(self):
        """Create temporary test images with different quality levels"""
        test_dir = Path(tempfile.mkdtemp())
        
        # High quality: smooth gradient
        img_hq = Image.new('RGB', (400, 300))
        pixels_hq = []
        for y in range(300):
            for x in range(400):
                r = int(255 * (x / 400))
                g = int(255 * (y / 300))
                b = 128
                pixels_hq.append((r, g, b))
        img_hq.putdata(pixels_hq)
        hq_path = test_dir / "high_quality.jpg"
        img_hq.save(str(hq_path))
        
        # Low quality: noisy image
        img_lq = Image.new('RGB', (200, 150))
        pixels_lq = []
        np.random.seed(42)  # For reproducible results
        for y in range(150):
            for x in range(200):
                noise = np.random.randint(-50, 50)
                r = np.clip(128 + noise, 0, 255)
                g = np.clip(64 + noise, 0, 255)
                b = np.clip(192 + noise, 0, 255)
                pixels_lq.append((r, g, b))
        img_lq.putdata(pixels_lq)
        lq_path = test_dir / "low_quality.jpg"
        img_lq.save(str(lq_path))
        
        yield {
            'high_quality': str(hq_path),
            'low_quality': str(lq_path),
            'test_dir': test_dir
        }
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_brisque_initialization(self, brisque_model):
        """Test BRISQUE model initialization"""
        assert brisque_model is not None
        assert hasattr(brisque_model, 'mu_prisparam')
        assert hasattr(brisque_model, 'cov_prisparam')
        assert len(brisque_model.mu_prisparam) == 18
    
    def test_calculate_brisque_score(self, brisque_model, test_images):
        """Test BRISQUE score calculation"""
        hq_score = brisque_model.calculate_brisque_score(test_images['high_quality'])
        lq_score = brisque_model.calculate_brisque_score(test_images['low_quality'])
        
        # Scores should be numeric
        assert isinstance(hq_score, float)
        assert isinstance(lq_score, float)
        
        # Scores should be in reasonable range
        assert 0 <= hq_score <= 100
        assert 0 <= lq_score <= 100
        
        # Low quality should have higher score (worse quality)
        assert lq_score > hq_score
    
    def test_normalize_score(self, brisque_model):
        """Test score normalization"""
        # Test various score values
        test_scores = [0, 20, 50, 80, 100]
        
        for score in test_scores:
            normalized = brisque_model.normalize_score(score)
            assert isinstance(normalized, float)
            assert 0 <= normalized <= 1
        
        # Lower raw scores should give higher normalized scores
        norm_low = brisque_model.normalize_score(10)
        norm_high = brisque_model.normalize_score(80)
        assert norm_low > norm_high
    
    def test_quality_description(self, brisque_model):
        """Test quality description mapping"""
        descriptions = [
            (5, "Excellent"),
            (25, "Good"),
            (45, "Fair"),
            (60, "Poor"),
            (85, "Very Poor")
        ]
        
        for score, expected_desc in descriptions:
            desc = brisque_model.get_quality_description(score)
            assert desc == expected_desc
    
    def test_invalid_image_path(self, brisque_model):
        """Test handling of invalid image paths"""
        score = brisque_model.calculate_brisque_score("nonexistent_image.jpg")
        # Should return default score on error
        assert score == 50.0
    
    def test_feature_extraction_methods(self, brisque_model, test_images):
        """Test internal feature extraction methods"""
        # Load test image for feature extraction
        import cv2
        image = cv2.imread(test_images['high_quality'], cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Test MSCN transform
        features = brisque_model._compute_image_mscn_transform(gray)
        assert isinstance(features, list)
        assert len(features) == 10  # 2 + 4*2 features per scale
        
        # Test full feature extraction
        all_features = brisque_model._extract_brisque_features(gray)
        assert isinstance(all_features, np.ndarray)
        assert len(all_features) == 20  # 10 features per scale, 2 scales


def test_brisque_standalone():
    """Standalone test function for backwards compatibility"""
    print("Testing BRISQUE model...")
    
    # Create test images
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # High quality: smooth gradient
        img_hq = Image.new('RGB', (400, 300))
        pixels_hq = []
        for y in range(300):
            for x in range(400):
                r = int(255 * (x / 400))
                g = int(255 * (y / 300))
                b = 128
                pixels_hq.append((r, g, b))
        img_hq.putdata(pixels_hq)
        hq_path = test_dir / "high_quality.jpg"
        img_hq.save(str(hq_path))
        
        # Low quality: noisy image
        img_lq = Image.new('RGB', (200, 150))
        pixels_lq = []
        np.random.seed(42)
        for y in range(150):
            for x in range(200):
                noise = np.random.randint(-50, 50)
                r = np.clip(128 + noise, 0, 255)
                g = np.clip(64 + noise, 0, 255)
                b = np.clip(192 + noise, 0, 255)
                pixels_lq.append((r, g, b))
        img_lq.putdata(pixels_lq)
        lq_path = test_dir / "low_quality.jpg"
        img_lq.save(str(lq_path))
        
        # Test BRISQUE model
        brisque_model = OpenCVBRISQUEModel()
        
        # Test high quality image
        hq_score = brisque_model.calculate_brisque_score(str(hq_path))
        hq_norm = brisque_model.normalize_score(hq_score)
        hq_desc = brisque_model.get_quality_description(hq_score)
        
        print(f"\nHigh Quality Image:")
        print(f"  Raw score: {hq_score:.2f}")
        print(f"  Normalized: {hq_norm:.3f}")
        print(f"  Description: {hq_desc}")
        
        # Test low quality image
        lq_score = brisque_model.calculate_brisque_score(str(lq_path))
        lq_norm = brisque_model.normalize_score(lq_score)
        lq_desc = brisque_model.get_quality_description(lq_score)
        
        print(f"\nLow Quality Image:")
        print(f"  Raw score: {lq_score:.2f}")
        print(f"  Normalized: {lq_norm:.3f}")
        print(f"  Description: {lq_desc}")
        
        print(f"\n✅ BRISQUE test completed!")
        print(f"High quality score: {hq_score:.1f} vs Low quality score: {lq_score:.1f}")
        
        if lq_score > hq_score:
            print("✅ BRISQUE correctly identified quality difference!")
        else:
            print("⚠️  BRISQUE results may need tuning")
            
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_brisque_standalone()