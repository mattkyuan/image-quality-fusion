# Deployment Guide

Simple guide for deploying your trained Image Quality Fusion model to Hugging Face Hub.

## Prerequisites

1. **Hugging Face Account**: Create account at https://huggingface.co
2. **Trained Model**: You need a model in `outputs/fixed_run/model_best.pth`
3. **Dependencies**: Install required packages

```bash
pip install huggingface_hub torch torchvision open-clip-torch opencv-python pillow
```

## Quick Deployment

### 1. Login to Hugging Face

```bash
huggingface-cli login
```

Enter your HF token when prompted (get it from https://huggingface.co/settings/tokens).

### 2. Deploy Your Model

```bash
# Simple deployment
python deploy_to_hf.py

# Test without uploading
python deploy_to_hf.py --dry-run

# Custom repository name
python deploy_to_hf.py --repo-name my-custom-model
```

### 3. Verify Deployment

The script will show you the URL of your deployed model. Visit it to test:

```
🎉 Deployment Complete!
🌐 Your model: https://huggingface.co/yourusername/image-quality-fusion
```

## What Gets Deployed

The deployment script automatically:

- ✅ **Converts your PyTorch model** to HF format
- ✅ **Creates repository** (if it doesn't exist)
- ✅ **Generates model card** with performance metrics
- ✅ **Uploads model files** (pytorch_model.bin, config.json)
- ✅ **Updates documentation** (README.md)

## Testing Your Deployed Model

```python
from huggingface_hub import PyTorchModelHubMixin

# Load your deployed model
model = PyTorchModelHubMixin.from_pretrained("yourusername/image-quality-fusion")

# Test with an image
quality_score = model.predict_quality("path/to/image.jpg")
print(f"Quality: {quality_score:.2f}/10")
```

Or use the test script:

```bash
python scripts/monitoring/simple_hf_test.py
```

## Troubleshooting

### Authentication Issues
```
❌ Authentication failed: Invalid user token
```
**Solution**: Run `huggingface-cli login` and enter a valid token.

### Model Not Found
```
❌ Model not found: outputs/fixed_run/model_best.pth
```
**Solution**: Train your model first:
```bash
python src/image_quality_fusion/training/train_fusion.py \
    --image_dir ./datasets/demo/images \
    --annotations ./datasets/demo/annotations.csv \
    --prepare_data --epochs 50
```

### Repository Creation Failed
```
❌ Failed to create repository
```
**Solution**: Check that your token has "Write" permissions and the repository name is available.

## Advanced Options

### Custom Repository Settings

```bash
# Deploy to specific repository
python deploy_to_hf.py --repo-name my-quality-model

# The script will create: yourusername/my-quality-model
```

### Manual Model Card Updates

If you want to customize the model card, edit it after deployment at:
`https://huggingface.co/yourusername/image-quality-fusion/edit/main/README.md`

## Repository Structure on HF Hub

After deployment, your HF repository will contain:

```
yourusername/image-quality-fusion/
├── README.md              # Auto-generated model card
├── config.json            # Model configuration  
├── pytorch_model.bin      # Your trained model weights
└── .gitattributes         # HF-specific git settings
```

## Next Steps

1. **Test your model** in the HF Hub interface
2. **Share the URL** with others to use your model
3. **Update the model** by running the deployment script again
4. **Monitor usage** through your HF dashboard

Your model is now publicly available and can be used by anyone with the simple HF Hub API!