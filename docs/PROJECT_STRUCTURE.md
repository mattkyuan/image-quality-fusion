# Project Structure Guide

This project is organized to support two main goals:
1. **Training**: Model development and training
2. **HF Deployment**: Hugging Face Hub deployment

## Key Files by Goal

### Goal 1: Training
**Essential:**
- `src/image_quality_fusion/` - Core package
- `scripts/training/` - Training scripts  
- `configs/` - Training configurations
- `datasets/` - Training data

### Goal 2: HF Deployment  
**Essential:**
- `deploy_to_hf.py` - Simple local deployment script
- `src/image_quality_fusion/models/huggingface_wrapper.py` - HF integration
- `scripts/monitoring/` - Testing scripts
- `outputs/fixed_run/` - Trained model
- `README.md`, `MODEL_CARD.md` - Documentation

## Path Management Architecture

The project uses a **portable relative path system** designed for maximum compatibility across different environments and development setups.

### Core Philosophy
- **No hardcoded absolute paths**: All paths are resolved relative to the project root
- **Automatic project root detection**: Uses marker files (src/, README.md, .git) to find the project base
- **Centralized path utilities**: Single source of truth for path resolution (`src/image_quality_fusion/utils/paths.py`)
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux

### Benefits
- **Portability**: Works from any directory within the project
- **Development Flexibility**: No need to be in specific directory to run scripts
- **CI/CD Compatibility**: Robust across different execution environments
- **Team Collaboration**: No user-specific path configurations

## Maintenance

### What's Gitignored
- Large datasets (`datasets/spaq/images/`, `datasets/processed/`)
- Training artifacts (`training_data/`, `outputs/` except fixed_run)
- Cache files (`*.npy`, `features.csv`)
- Virtual environments (`.venv/`)

### What's Included in Git
- All source code (`src/`)
- Essential scripts (`scripts/`)
- Configuration files (`configs/`)
- Documentation (`docs/`, `*.md`)
- Successful training results (`outputs/fixed_run/`)
- Simple deployment script (`deploy_to_hf.py`)
- Dataset metadata (CSV files, not images)

This structure supports rapid iteration while maintaining clean separation between development, training, and deployment concerns.