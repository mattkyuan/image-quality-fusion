# src/image_quality_fusion/tests/test_fusion_training.py
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image

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

from ..models.fusion_model import ImageQualityFusionModel, WeightedFusionModel
from ..training.trainer import ImageQualityDataset, FusionModelTrainer
from ..training.data_utils import create_sample_annotations, prepare_training_data


class TestFusionModels:
    """Test suite for fusion models"""
    
    @pytest.fixture
    def test_data_setup(self):
        """Create test data setup"""
        test_dir = Path(tempfile.mkdtemp())
        
        # Create sample images
        image_dir = test_dir / "images"
        image_dir.mkdir()
        
        # Create 10 test images
        image_paths = []
        for i in range(10):
            img = Image.new('RGB', (224, 224), color=(i*25, 100, 150))
            img_path = image_dir / f"test_image_{i:02d}.jpg"
            img.save(str(img_path))
            image_paths.append(str(img_path.relative_to(test_dir)))
        
        # Create sample annotations
        annotations_path = test_dir / "annotations.csv"
        annotations = pd.DataFrame({
            'image_path': image_paths,
            'human_score': np.random.uniform(1.0, 10.0, size=10)
        })
        annotations.to_csv(annotations_path, index=False)
        
        yield {
            'test_dir': test_dir,
            'image_dir': str(image_dir),
            'annotations_path': str(annotations_path),
            'image_paths': image_paths
        }
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_fusion_model_creation(self):
        """Test fusion model initialization"""
        model = ImageQualityFusionModel(
            clip_embed_dim=512,
            hidden_dim=256,
            dropout_rate=0.3,
            output_range=(1.0, 10.0)
        )
        
        assert model is not None
        assert model.clip_embed_dim == 512
        assert model.hidden_dim == 256
        assert model.output_range == (1.0, 10.0)
        
        # Test parameter count
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        print(f"Model parameters: {param_count:,}")
    
    def test_fusion_model_forward(self):
        """Test fusion model forward pass"""
        model = ImageQualityFusionModel()
        model.eval()
        
        # Create sample input
        batch_size = 4
        features = {
            'brisque': torch.randn(batch_size, 1),
            'laion': torch.randn(batch_size, 1),
            'clip': torch.randn(batch_size, 512)
        }
        
        # Forward pass
        with torch.no_grad():
            output = model(features)
        
        # Check output shape and range
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 1.0) and torch.all(output <= 10.0)
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    def test_weighted_fusion_model(self):
        """Test weighted fusion model"""
        model = WeightedFusionModel(output_range=(1.0, 10.0))
        model.eval()
        
        batch_size = 4
        features = {
            'brisque': torch.randn(batch_size, 1),
            'laion': torch.randn(batch_size, 1),
            'clip': torch.randn(batch_size, 512)
        }
        
        with torch.no_grad():
            output = model(features)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 1.0) and torch.all(output <= 10.0)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create model
        original_model = ImageQualityFusionModel(
            hidden_dim=128,
            output_range=(2.0, 8.0)
        )
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Save model
            metadata = {'test': 'data', 'epoch': 42}
            original_model.save_model(temp_path, metadata)
            
            # Load model
            loaded_model, loaded_metadata = ImageQualityFusionModel.load_model(temp_path)
            
            # Check model configuration
            assert loaded_model.hidden_dim == 128
            assert loaded_model.output_range == (2.0, 8.0)
            
            # Check metadata
            assert loaded_metadata['test'] == 'data'
            assert loaded_metadata['epoch'] == 42
            
            # Test that models produce same output
            test_features = {
                'brisque': torch.tensor([[0.5]]),
                'laion': torch.tensor([[0.7]]),
                'clip': torch.randn(1, 512)
            }
            
            original_model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                original_output = original_model(test_features)
                loaded_output = loaded_model(test_features)
            
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
            
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
    
    def test_dataset_creation(self, test_data_setup):
        """Test dataset creation"""
        # Prepare features
        features_path, embeddings_path = prepare_training_data(
            image_dir=test_data_setup['image_dir'],
            annotations_path=test_data_setup['annotations_path'],
            output_dir=str(test_data_setup['test_dir'] / 'features')
        )
        
        # Create dataset
        dataset = ImageQualityDataset(
            annotations_path=test_data_setup['annotations_path'],
            features_path=features_path,
            embeddings_path=embeddings_path
        )
        
        assert len(dataset) == 10
        
        # Test sample retrieval
        features, target = dataset[0]
        
        assert 'brisque' in features
        assert 'laion' in features
        assert 'clip' in features
        
        assert features['brisque'].shape == (1, 1)
        assert features['laion'].shape == (1, 1)
        assert features['clip'].shape == (1, 512)
        assert target.shape == (1,)
        
        # Check value ranges
        assert 0 <= features['brisque'].item() <= 1
        assert 0 <= features['laion'].item() <= 1
        assert 1 <= target.item() <= 10
    
    def test_mini_training_loop(self, test_data_setup):
        """Test a minimal training loop"""
        # Prepare features
        features_path, embeddings_path = prepare_training_data(
            image_dir=test_data_setup['image_dir'],
            annotations_path=test_data_setup['annotations_path'],
            output_dir=str(test_data_setup['test_dir'] / 'features')
        )
        
        # Create dataset (use same for train/val for testing)
        dataset = ImageQualityDataset(
            annotations_path=test_data_setup['annotations_path'],
            features_path=features_path,
            embeddings_path=embeddings_path
        )
        
        # Create small model
        model = ImageQualityFusionModel(
            hidden_dim=64,
            dropout_rate=0.1
        )
        
        # Create trainer
        output_dir = test_data_setup['test_dir'] / 'training_output'
        trainer = FusionModelTrainer(
            model=model,
            device='cpu',
            output_dir=str(output_dir)
        )
        
        # Run minimal training
        history = trainer.train(
            train_dataset=dataset,
            val_dataset=dataset,  # Using same dataset for testing
            batch_size=2,
            epochs=3,
            learning_rate=1e-2,
            patience=5
        )
        
        # Check training history
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) <= 3  # May stop early
        assert len(history['val_losses']) <= 3
        
        # Check metrics
        assert 'final_metrics' in history
        assert 'mse' in history['final_metrics']
        assert 'mae' in history['final_metrics']
        assert 'r2' in history['final_metrics']
        
        print(f"Training completed - Final MAE: {history['final_metrics']['mae']:.4f}")
        
        # Check that model files were saved
        assert (output_dir / 'model_best.pth').exists()


def test_fusion_training_standalone():
    """Standalone test function for fusion training"""
    print("Testing Fusion Model Training...")
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create sample images
        image_dir = test_dir / "images"
        image_dir.mkdir()
        
        print("Creating test images...")
        for i in range(20):
            # Create varied test images
            img = Image.new('RGB', (300, 200))
            pixels = []
            for y in range(200):
                for x in range(300):
                    r = int(255 * (x / 300)) if i % 2 == 0 else int(255 * (y / 200))
                    g = int(255 * (y / 200)) if i % 3 == 0 else 128
                    b = 128 + i * 5
                    pixels.append((r, g, b))
            img.putdata(pixels)
            img.save(str(image_dir / f"test_{i:02d}.jpg"))
        
        # Create annotations
        annotations_path = test_dir / "annotations.csv"
        np.random.seed(42)
        annotations = pd.DataFrame({
            'image_path': [f"test_{i:02d}.jpg" for i in range(20)],
            'human_score': np.random.uniform(3.0, 9.0, size=20)
        })
        annotations.to_csv(annotations_path, index=False)
        print(f"Created {len(annotations)} annotations")
        
        # Test feature extraction
        print("Extracting features...")
        features_path, embeddings_path = prepare_training_data(
            image_dir=str(image_dir),
            annotations_path=str(annotations_path),
            output_dir=str(test_dir / 'features'),
            batch_size=5
        )
        
        # Create dataset
        print("Creating dataset...")
        dataset = ImageQualityDataset(
            annotations_path=str(annotations_path),
            features_path=features_path,
            embeddings_path=embeddings_path
        )
        print(f"Dataset size: {len(dataset)}")
        
        # Test different model types
        model_types = [
            ("Deep Fusion", ImageQualityFusionModel(hidden_dim=64)),
            ("Weighted Fusion", WeightedFusionModel())
        ]
        
        for model_name, model in model_types:
            print(f"\nTesting {model_name}...")
            
            # Create trainer
            output_dir = test_dir / f'training_{model_name.lower().replace(" ", "_")}'
            trainer = FusionModelTrainer(
                model=model,
                device='cpu',
                output_dir=str(output_dir)
            )
            
            # Mini training run
            history = trainer.train(
                train_dataset=dataset,
                val_dataset=dataset,
                batch_size=4,
                epochs=5,
                learning_rate=1e-2,
                patience=10
            )
            
            print(f"  Final metrics:")
            for metric, value in history['final_metrics'].items():
                print(f"    {metric.upper()}: {value:.4f}")
            
            # Test model save/load
            model_path = output_dir / 'test_model.pth'
            model.save_model(str(model_path))
            
            loaded_model, metadata = type(model).load_model(str(model_path), 'cpu')
            print(f"  Model saved and loaded successfully")
        
        print("\n✅ Fusion training test completed successfully!")
        
    except Exception as e:
        print(f"❌ Fusion training test failed: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_fusion_training_standalone()