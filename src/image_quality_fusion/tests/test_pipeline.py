# src/image_quality_fusion/tests/test_pipeline.py
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
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

from ..data.preprocessing import ImageQualityExtractor
from ..models.brisque_opencv import OpenCVBRISQUEModel
from ..models.aesthetic_predictor_original import LAIONAestheticsModel
from ..models.clip_model import CLIPSemanticAnalyzer


class TestImageQualityPipeline:
    """Test suite for full image quality pipeline"""
    
    @pytest.fixture
    def extractor(self):
        """Create ImageQualityExtractor instance"""
        return ImageQualityExtractor()
    
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
            'paths': [str(sharp_path), str(simple_path), str(checker_path)],
            'test_dir': test_dir
        }
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_extractor_initialization(self, extractor):
        """Test that extractor initializes all models correctly"""
        assert extractor is not None
        assert isinstance(extractor.brisque_model, OpenCVBRISQUEModel)
        assert isinstance(extractor.laion_model, LAIONAestheticsModel)
        assert isinstance(extractor.clip_model, CLIPSemanticAnalyzer)
    
    def test_single_image_extraction(self, extractor, test_images):
        """Test feature extraction for a single image"""
        image_path = test_images['paths'][0]
        
        features = extractor.extract_features_single_image(image_path)
        
        # Check that features were extracted
        assert features is not None
        assert isinstance(features, dict)
        
        # Check required keys
        required_keys = [
            'image_path', 'brisque_raw', 'aesthetic_raw',
            'brisque_normalized', 'aesthetic_normalized',
            'clip_embedding', 'technical_quality', 'aesthetic_quality',
            'embedding_dim'
        ]
        
        for key in required_keys:
            assert key in features
        
        # Check data types and ranges
        assert isinstance(features['brisque_raw'], float)
        assert isinstance(features['aesthetic_raw'], float)
        assert isinstance(features['brisque_normalized'], float)
        assert isinstance(features['aesthetic_normalized'], float)
        assert isinstance(features['clip_embedding'], np.ndarray)
        assert isinstance(features['embedding_dim'], int)
        
        # Check normalized scores are in [0, 1]
        assert 0 <= features['brisque_normalized'] <= 1
        assert 0 <= features['aesthetic_normalized'] <= 1
        
        # Check raw scores are in expected ranges
        assert 0 <= features['brisque_raw'] <= 100
        assert 1 <= features['aesthetic_raw'] <= 10
        
        # Check embedding properties
        assert len(features['clip_embedding']) == features['embedding_dim']
        assert features['embedding_dim'] > 0
    
    def test_batch_extraction(self, extractor, test_images):
        """Test batch feature extraction"""
        image_paths = test_images['paths']
        
        # Test without saving
        df = extractor.extract_features_batch(image_paths)
        
        # Check DataFrame properties
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(image_paths)
        
        # Check columns
        expected_columns = [
            'image_path', 'brisque_raw', 'aesthetic_raw',
            'brisque_normalized', 'aesthetic_normalized',
            'clip_embedding', 'technical_quality', 'aesthetic_quality',
            'embedding_dim'
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check that all images were processed
        processed_paths = set(df['image_path'].values)
        expected_paths = set(image_paths)
        assert processed_paths == expected_paths
    
    def test_batch_extraction_with_save(self, extractor, test_images):
        """Test batch extraction with file saving"""
        image_paths = test_images['paths']
        output_path = test_images['test_dir'] / "results.csv"
        
        # Test with saving
        df = extractor.extract_features_batch(image_paths, str(output_path))
        
        # Check that files were created
        assert output_path.exists()
        embeddings_path = output_path.parent / "results_embeddings.npy"
        assert embeddings_path.exists()
        
        # Check saved CSV
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(image_paths)
        
        # Check saved embeddings
        saved_embeddings = np.load(embeddings_path)
        assert saved_embeddings.shape[0] == len(image_paths)
        assert saved_embeddings.shape[1] > 0
    
    def test_invalid_image_handling(self, extractor):
        """Test handling of invalid images"""
        # Test with non-existent file
        features = extractor.extract_features_single_image("nonexistent.jpg")
        assert features is None
        
        # Test batch with mix of valid and invalid paths
        test_dir = Path(tempfile.mkdtemp())
        try:
            # Create one valid image
            valid_img = Image.new('RGB', (100, 100), color=(255, 0, 0))
            valid_path = test_dir / "valid.jpg"
            valid_img.save(str(valid_path))
            
            # Mix valid and invalid paths
            mixed_paths = [str(valid_path), "invalid1.jpg", "invalid2.jpg"]
            
            df = extractor.extract_features_batch(mixed_paths)
            
            # Should only contain the valid image
            assert len(df) == 1
            assert df.iloc[0]['image_path'] == str(valid_path)
            
        finally:
            shutil.rmtree(test_dir)
    
    def test_different_image_formats(self, extractor):
        """Test extraction with different image formats and sizes"""
        test_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create images with different properties
            images = {
                'small.jpg': Image.new('RGB', (50, 50), color=(255, 0, 0)),
                'large.jpg': Image.new('RGB', (1000, 800), color=(0, 255, 0)),
                'grayscale.jpg': Image.new('L', (200, 200), color=128),
                'square.jpg': Image.new('RGB', (300, 300), color=(0, 0, 255))
            }
            
            paths = []
            for filename, img in images.items():
                path = test_dir / filename
                # Convert grayscale to RGB for consistency
                if img.mode == 'L':
                    img = img.convert('RGB')
                img.save(str(path))
                paths.append(str(path))
            
            # Extract features for all images
            df = extractor.extract_features_batch(paths)
            
            # Check that all images were processed
            assert len(df) == len(paths)
            
            # Check that embeddings have consistent dimensions
            embedding_dims = df['embedding_dim'].unique()
            assert len(embedding_dims) == 1  # All should have same dimension
            
        finally:
            shutil.rmtree(test_dir)


def test_basic_pipeline_standalone():
    """Standalone test function for backwards compatibility"""
    print("=" * 60)
    print("IMAGE QUALITY FUSION - BASIC PIPELINE TEST")
    print("=" * 60)
    
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
        print(f"Created {len(test_files)} test images")
        
        # Track results
        results = []
        
        # Test 1: Individual model components
        print("\n1. Testing individual model components...")
        
        try:
            brisque_model = OpenCVBRISQUEModel()
            laion_model = LAIONAestheticsModel()
            clip_model = CLIPSemanticAnalyzer()
            print("   ✅ All models initialized successfully")
            results.append(("Model Initialization", True))
        except Exception as e:
            print(f"   ❌ Model initialization failed: {e}")
            results.append(("Model Initialization", False))
            return
        
        # Test 2: Full pipeline extraction
        print("\n2. Testing full pipeline extraction...")
        
        try:
            extractor = ImageQualityExtractor()
            
            # Test single image
            features = extractor.extract_features_single_image(test_files[0])
            if features:
                print(f"   ✅ Single image extraction successful")
                print(f"      BRISQUE: {features['brisque_raw']:.2f} -> {features['brisque_normalized']:.3f}")
                print(f"      Aesthetic: {features['aesthetic_raw']:.2f} -> {features['aesthetic_normalized']:.3f}")
                print(f"      Embedding dim: {features['embedding_dim']}")
                results.append(("Single Image Extraction", True))
            else:
                print("   ❌ Single image extraction failed")
                results.append(("Single Image Extraction", False))
        except Exception as e:
            print(f"   ❌ Single image extraction failed: {e}")
            results.append(("Single Image Extraction", False))
        
        # Test 3: Batch processing
        print("\n3. Testing batch processing...")
        
        try:
            df = extractor.extract_features_batch(test_files)
            print(f"   ✅ Batch processing successful")
            print(f"      Processed {len(df)} images")
            print(f"      Columns: {list(df.columns)}")
            results.append(("Batch Processing", True))
        except Exception as e:
            print(f"   ❌ Batch processing failed: {e}")
            results.append(("Batch Processing", False))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:<25}: {status}")
        
        all_passed = all(result[1] for result in results)
        
        if all_passed:
            print(f"\n🎉 All tests passed! Pipeline is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Check the error messages above.")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        print(f"\nCleaning up test images...")
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_basic_pipeline_standalone()