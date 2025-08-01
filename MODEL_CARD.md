---
license: mit
tags:
- image-quality
- computer-vision
- brisque
- aesthetic
- clip
- fusion
- image-assessment
language:
- en
pipeline_tag: image-classification
library_name: pytorch
datasets:
- spaq
---

# Image Quality Fusion Model

A multi-modal image quality assessment system that combines BRISQUE, Aesthetic Predictor, and CLIP features to predict human-like quality judgments.

## Model Description

This model fuses three complementary approaches to image quality assessment:

- **🔧 BRISQUE (OpenCV)**: Technical quality assessment detecting blur, noise, and compression artifacts
- **🎨 Aesthetic Predictor**: Visual appeal assessment using LAION-trained CLIP features  
- **🧠 CLIP**: Semantic understanding and feature extraction (ViT-B-32)

The fusion network learns to combine these diverse signals to predict human quality judgments on a 1-10 scale.

## Performance

Trained and evaluated on the SPAQ dataset (11,125 smartphone images):

| Metric | Value |
|--------|-------|
| **Correlation with humans** | 0.520 |
| **R² Score** | 0.250 |
| **Mean Absolute Error** | 1.41 points |
| **Mean Squared Error** | 2.86 |

## Usage

### Quick Start

```python
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

# Load model
model = PyTorchModelHubMixin.from_pretrained("YOUR_USERNAME/image-quality-fusion")

# Predict quality for an image
quality_score = model.predict_quality("path/to/your/image.jpg")
print(f"Quality score: {quality_score:.2f}/10")

# Batch prediction
scores = model.predict_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
```

### Advanced Usage

```python
from PIL import Image
import numpy as np

# Load image
image = Image.open("photo.jpg")

# Get quality score
score = model.predict_quality(image)

# The model accepts multiple input formats:
# - File paths (str)
# - PIL Images
# - NumPy arrays

# Model also provides detailed feature information
model_info = model.get_model_info()
print(f"Model version: {model_info['version']}")
```

## Model Architecture

```
Input Image
    ├── OpenCV BRISQUE → Technical Quality Score (0-100, normalized)
    ├── Aesthetic Predictor → Aesthetic Score (0-10, normalized) 
    └── CLIP ViT-B-32 → Semantic Features (512-dim)
                ↓
        Deep Fusion Network
        (256 hidden units, dropout 0.3)
                ↓
        Human-like Quality Score (1-10)
```

The fusion network is a feed-forward neural network that learns optimal weights for combining the three feature types.

## Training Details

### Training Data
- **Dataset**: SPAQ (Smartphone Photography Attribute and Quality)
- **Images**: 11,125 smartphone photos
- **Annotations**: Human quality ratings (1-10 scale)
- **Split**: 80% train, 10% validation, 10% test

### Training Configuration
- **Batch Size**: 128 (optimized for M1 MacBook Pro)
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-3 with ReduceLROnPlateau scheduling
- **Loss Function**: MSE Loss
- **Regularization**: Dropout (0.3), Early stopping (patience=10)
- **Hardware**: M1 MacBook Pro with MPS acceleration
- **Training Time**: ~1 hour (with feature caching)

### Optimizations
- **Mixed Precision Training**: MPS autocast for M1 optimization
- **Feature Caching**: Pre-computed features for 20-30x speedup
- **Intelligent DataLoader**: 6-8 workers optimized for unified memory
- **Memory Management**: Garbage collection every 10 batches

## Limitations and Biases

### Limitations
- Trained specifically on smartphone photography - may not generalize to other image types
- BRISQUE component may not capture all technical quality aspects
- Aesthetic preferences can be subjective and culturally biased
- Model size requires ~100MB for inference

### Biases
- Training data bias toward smartphone photography aesthetic preferences
- Potential cultural bias in human annotations from SPAQ dataset
- CLIP model inherits biases from its training data

### Recommendations
- Test on your specific domain before production use
- Consider fine-tuning on domain-specific data
- Use as one component in a broader quality assessment pipeline

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{image-quality-fusion-2024,
  title={Image Quality Fusion: Multi-Modal Assessment of Image Quality},
  author={Your Name},
  year={2024},
  howpublished={\url{https://huggingface.co/YOUR_USERNAME/image-quality-fusion}}
}
```

## Acknowledgments

- [BRISQUE Implementation](https://learnopencv.com/image-quality-assessment-brisque/) by OpenCV
- [Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor) by LAION-AI  
- [OpenCLIP](https://github.com/mlfoundations/open_clip) by ML Foundations
- [SPAQ Dataset](https://github.com/h4nwei/SPAQ) for training and evaluation

## License

MIT License - see LICENSE file for details.

## Model Card Contact

For questions about this model, please open an issue in the [source repository](https://github.com/YOUR_USERNAME/image-quality-fusion).
