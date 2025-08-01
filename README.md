# Image Quality Fusion

A multi-modal image quality assessment system that fuses BRISQUE, Aesthetic, and CLIP features to predict human quality judgments.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project combines three complementary approaches to image quality assessment:

- **🔧 BRISQUE (OpenCV)**: Technical quality assessment (blur, noise, artifacts)
- **🎨 Aesthetic Predictor**: Visual appeal assessment via LAION-trained CLIP features  
- **🧠 CLIP**: Semantic understanding and feature extraction

The fusion model learns to combine these diverse signals to predict human quality judgments, achieving stronger performance than individual metrics.

## 🏗️ Architecture

```
Input Image
    ├── OpenCV BRISQUE → Technical Quality Score (0-100)
    ├── Aesthetic Predictor → Aesthetic Score (0-10) 
    └── CLIP Model → Semantic Features (512-dim)
                ↓
        Deep Fusion Network
                ↓
        Human-like Quality Score (1-10)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-quality-fusion.git
cd image-quality-fusion

# Install dependencies
pip install -e .
```

### Basic Usage

```python
from src.image_quality_fusion.models.fusion_model import ImageQualityFusionModel
from src.image_quality_fusion.data.preprocessing import ImageQualityExtractor

# Extract features
extractor = ImageQualityExtractor()
features = extractor.extract_features_single_image("path/to/image.jpg")

# Load trained model (you'll need to train first)
model, metadata = ImageQualityFusionModel.load_model("model.pth")
quality_score = model.predict(features)

print(f"Predicted quality score: {quality_score:.2f}/10")
```

### Training on Your Data

```python
# Prepare your dataset (CSV with image_path, human_score columns)
python src/image_quality_fusion/training/train_fusion.py \
    --image_dir data/images \
    --annotations data/annotations.csv \
    --prepare_data \
    --model_type deep \
    --epochs 100 \
    --output_dir experiments/my_training
```

## 📊 Performance

Trained on SPAQ dataset (11,125 smartphone images):
- **Correlation with humans**: 0.44
- **R² Score**: 0.16  
- **Mean Absolute Error**: 1.5 points (on 1-10 scale)

*Note: Performance varies significantly based on training data quality and size.*

## 🏗️ Project Structure

```
image-quality-fusion/
├── src/image_quality_fusion/
│   ├── data/              # Data preprocessing
│   ├── models/            # Model implementations  
│   ├── training/          # Training pipeline
│   └── utils/             # Helper utilities
├── configs/               # Training configurations
├── scripts/               # High-level scripts
├── datasets/              
│   └── demo/              # Small demo dataset
└── tests/                 # Unit tests
```

## 🔬 Components

### BRISQUE (Technical Quality)
- No-reference image quality assessment
- Detects blur, noise, compression artifacts
- Fast computation using OpenCV implementation

### Aesthetic Predictor
- Based on LAION's aesthetic predictor
- Uses CLIP ViT-B-32 features
- Trained on human aesthetic ratings

### CLIP Integration  
- ViT-B-32 or ViT-L-14 models supported
- Provides semantic understanding
- 512/768-dimensional embeddings

### Fusion Models
- **Linear**: Simple weighted combination
- **Weighted**: Learned feature weights
- **Deep**: Multi-layer neural network (recommended)

## 📈 Training Your Own Model

### 1. Prepare Data
Your CSV should have columns: `image_path`, `human_score`

```csv
image_path,human_score
images/photo1.jpg,7.2
images/photo2.jpg,4.8
```

### 2. Extract Features
```bash
# This extracts BRISQUE, Aesthetic, and CLIP features
python scripts/extract_features.py --image_dir data/images --annotations data.csv
```

### 3. Train Model
```bash
python scripts/train_model.py --config configs/default.yaml
```

## 🛠️ Advanced Usage

### Custom Feature Extraction
```python
from src.image_quality_fusion.models.brisque_opencv import OpenCVBRISQUEModel
from src.image_quality_fusion.models.aesthetic_predictor_original import OriginalAestheticPredictor
from src.image_quality_fusion.models.clip_model import CLIPSemanticAnalyzer

# Initialize individual models
brisque = OpenCVBRISQUEModel()
aesthetic = OriginalAestheticPredictor()
clip = CLIPSemanticAnalyzer(model_name='ViT-B-32')

# Extract features
technical_score = brisque.calculate_brisque_score("image.jpg")
aesthetic_score = aesthetic.calculate_aesthetic_score("image.jpg") 
semantic_features = clip.encode_image("image.jpg")
```

### Model Architectures
```python
from src.image_quality_fusion.models.fusion_model import (
    ImageQualityFusionModel,  # Deep neural network
    WeightedFusionModel,      # Learned weights
    EnsembleFusionModel       # Multiple model ensemble
)

# Create different model types
deep_model = ImageQualityFusionModel(clip_embed_dim=512, hidden_dim=256)
weighted_model = WeightedFusionModel(output_range=(1.0, 10.0))
ensemble_model = EnsembleFusionModel(output_range=(1.0, 10.0))
```

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- OpenCV 4.5+
- open-clip-torch
- PIL/Pillow
- pandas, numpy, scikit-learn
- tqdm

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{image-quality-fusion,
  title={Image Quality Fusion: Multi-Modal Assessment of Image Quality},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/yourusername/image-quality-fusion}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BRISQUE Implementation](https://learnopencv.com/image-quality-assessment-brisque/) by OpenCV
- [Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor) by LAION-AI
- [OpenCLIP](https://github.com/mlfoundations/open_clip) by ML Foundations
- [SPAQ Dataset](https://github.com/h4nwei/SPAQ) for training and evaluation

## 🚧 Roadmap

- [ ] Add more sophisticated fusion architectures (attention, transformers)
- [ ] Support for additional quality metrics (LPIPS, SSIM, etc.)
- [ ] Web interface for easy testing
- [ ] Pre-trained models for common use cases
- [ ] Docker containerization
- [ ] Batch processing utilities

---

**Note**: This is a research project. For production use, consider additional validation on your specific domain and use case.