# Image Quality Fusion

A multi-modal image quality assessment system that fuses BRISQUE, Aesthetic, and CLIP features to predict human quality judgments.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/matthewyuan/image-quality-fusion)

## 🎯 Overview

This project combines three complementary approaches to image quality assessment:

- **🔧 BRISQUE (OpenCV)**: Technical quality assessment (blur, noise, artifacts)
- **🎨 Aesthetic Predictor**: Visual appeal assessment via LAION-trained CLIP features  
- **🧠 CLIP**: Semantic understanding and feature extraction

The fusion model learns to combine these diverse signals to predict human quality judgments, achieving stronger performance than individual metrics.

## ✨ Key Features

- **🎯 Ready to Use**: Pre-trained model available on Hugging Face Hub
- **🚀 Simple Deployment**: One-command deployment to HF Hub  
- **⚡ Optimized**: M1 MacBook Pro optimizations (75+ hours → ~1 hour training)
- **🔧 Portable**: Relative paths work across different systems
- **📊 Comprehensive**: Three complementary quality assessment approaches
- **🧪 Well-Tested**: 100% test coverage across all components

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
git clone https://github.com/mattkyuan/image-quality-fusion.git
cd image-quality-fusion

# Install dependencies
pip install torch torchvision torchaudio open-clip-torch opencv-python pillow pandas numpy scikit-learn tqdm matplotlib pytorch-lightning huggingface_hub
```

### Quick Test

Verify your installation works:

```python
from huggingface_hub import PyTorchModelHubMixin

# Test with pre-trained model
model = PyTorchModelHubMixin.from_pretrained("matthewyuan/image-quality-fusion")
print("✅ Installation successful!")
```

### Using the Pre-trained Model (Recommended)

```python
# Easy usage with Hugging Face Hub
from huggingface_hub import PyTorchModelHubMixin

# Load the pre-trained model
model = PyTorchModelHubMixin.from_pretrained("matthewyuan/image-quality-fusion")

# Predict quality for any image
quality_score = model.predict_quality("path/to/your/image.jpg")
print(f"Image quality: {quality_score:.2f}/10")

# Batch prediction
scores = model.predict_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
for i, score in enumerate(scores, 1):
    print(f"Image {i}: {score:.2f}/10")
```

### Local Development Usage

```python
from src.image_quality_fusion.models.fusion_model import ImageQualityFusionModel
from src.image_quality_fusion.data.preprocessing import ImageQualityExtractor
import torch

# Extract features for a single image
extractor = ImageQualityExtractor()
features = extractor.extract_features_single_image("path/to/image.jpg")

# Load a locally trained model
model, metadata = ImageQualityFusionModel.load_model("outputs/fixed_run/model_best.pth")

# Prepare features for model
features_tensor = {
    'brisque': torch.tensor([features['brisque_normalized']], dtype=torch.float32),
    'laion': torch.tensor([features['aesthetic_normalized']], dtype=torch.float32),
    'clip': torch.tensor(features['clip_embedding'], dtype=torch.float32).unsqueeze(0)
}

# Predict quality
quality_score = model.predict(features_tensor)
print(f"Predicted quality score: {quality_score.item():.2f}/10")
```

### Training on Your Data

```bash
# Quick training with optimized pipeline (M1 MacBook Pro optimized)
cd src && python -m image_quality_fusion.training.train_fusion \
    --image_dir ../datasets/demo/images \
    --annotations ../datasets/demo/annotations.csv \
    --prepare_data \
    --model_type deep \
    --batch_size 128 \
    --mixed_precision \
    --epochs 50 \
    --experiment_name my_model

# Or use the automated script
./scripts/training/run_training.sh
```

### 🚀 Deploy Your Model to Hugging Face

After training, deploy your model with one command:

```bash
# One-time setup
huggingface-cli login

# Deploy your trained model
python deploy_to_hf.py

# Test deployment (optional)
python deploy_to_hf.py --dry-run
```

📖 See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for detailed deployment instructions.

## 🔄 Complete Workflow

The typical workflow for using this project:

1. **🚀 Quick Start** - Use the pre-trained HF model (recommended for most users)
2. **🎯 Custom Training** - Train on your own data if needed  
3. **📤 Deploy** - Share your model via Hugging Face Hub
4. **🧪 Test** - Verify your deployment works

```bash
# Option 1: Use pre-trained model (recommended)
python -c "
from huggingface_hub import PyTorchModelHubMixin
model = PyTorchModelHubMixin.from_pretrained('matthewyuan/image-quality-fusion')
print(f'Quality: {model.predict_quality(\"path/to/image.jpg\"):.2f}/10')
"

# Option 2: Train your own model
cd src && python -m image_quality_fusion.training.train_fusion --help

# Option 3: Deploy your trained model
python deploy_to_hf.py
```

## 📊 Performance

Trained on SPAQ dataset (11,125 smartphone images) with optimized pipeline:
- **Correlation with humans**: 0.52
- **R² Score**: 0.25  
- **Mean Absolute Error**: 1.43 points (on 1-10 scale)
- **Training time**: ~1 minute (with cached features)

*Significant improvements achieved through M1 optimization and advanced caching.*

## 🏗️ Project Structure

```
image-quality-fusion/
├── src/image_quality_fusion/     # Core Python package
│   ├── data/                     # Data preprocessing & feature extraction
│   ├── models/                   # Model implementations & HF wrapper
│   ├── training/                 # Training pipeline
│   └── utils/                    # Shared utilities & path management
├── scripts/                      # Organized executable scripts
│   ├── training/                 # Training-related scripts
│   ├── deployment/               # HuggingFace deployment scripts
│   ├── monitoring/               # Testing and monitoring
│   └── utils/                    # Utility scripts
├── deploy_to_hf.py              # 🚀 Simple HF deployment script
├── configs/                     # Training configurations
├── docs/                        # Documentation
│   ├── PROJECT_STRUCTURE.md    # Project organization guide
│   └── DEPLOYMENT.md           # HF deployment instructions
├── outputs/fixed_run/           # Your trained models
└── datasets/                    # Training data (demo included)
```

The project uses **relative paths** throughout for maximum portability across different systems.

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

### 2. Train Model (Features Extracted Automatically)
```bash
# Full pipeline with M1 optimization
cd src && python -m image_quality_fusion.training.train_fusion \
    --image_dir ../datasets/demo/images \
    --annotations ../datasets/demo/annotations.csv \
    --prepare_data \
    --batch_size 128 \
    --mixed_precision \
    --epochs 50

# Monitor training progress (in another terminal)
python scripts/training/monitor_training.py
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
- huggingface_hub (for deployment)

## 🔧 Troubleshooting

### Common Issues

**Training Script Import Error:**
```bash
# ❌ This won't work
python src/image_quality_fusion/training/train_fusion.py

# ✅ Use this instead  
cd src && python -m image_quality_fusion.training.train_fusion
```

**Deployment Authentication:**
```bash
# If deploy_to_hf.py fails with auth error
huggingface-cli login
```

**Model Not Found:**
```bash
# Ensure you have a trained model first
ls outputs/fixed_run/model_best.pth
```

**Path Issues:**
The project uses relative paths - make sure you're running commands from the project root directory.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{image-quality-fusion,
  title={Image Quality Fusion: Multi-Modal Assessment of Image Quality},
  author={Matthew Yuan},
  year={2024},
  howpublished={\\url{https://github.com/mattkyuan/image-quality-fusion}}
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

## ⚡ M1 MacBook Pro Optimizations

This project includes specific optimizations for M1 MacBook Pro:
- **Mixed precision training** with MPS autocast
- **Optimized DataLoader** with 6-8 workers 
- **Intelligent feature caching** for 20-30x speedup
- **Memory management** for large datasets
- **Batch processing** optimized for 32GB unified memory

Training time reduced from 75+ hours to ~1 hour total!

## 🚧 Roadmap

**✅ Completed:**
- [x] M1 MacBook Pro optimization
- [x] Mixed precision training  
- [x] Advanced feature caching
- [x] Optimized training pipeline
- [x] Simple Hugging Face Hub deployment
- [x] Portable relative path architecture
- [x] Comprehensive test coverage
- [x] Clean project organization

**🔮 Future Enhancements:**
- [ ] Model export (ONNX, TorchScript)
- [ ] Web interface for easy testing
- [ ] Docker containerization  
- [ ] API endpoint deployment
- [ ] Additional dataset support
- [ ] Model ensemble techniques

---

**Note**: This is a research project. For production use, consider additional validation on your specific domain and use case.