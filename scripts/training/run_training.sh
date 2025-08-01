#!/bin/bash

# Training script with logging
echo "🚀 Starting optimized image quality fusion training..."
echo "📅 Started at: $(date)"

# Change to project root directory
cd "$(dirname "$0")/../.."

# Run training with output captured
python src/image_quality_fusion/training/train_fusion.py \
    --image_dir ./datasets/spaq/images \
    --annotations ./datasets/processed/spaq/annotations_fusion.csv \
    --prepare_data \
    --batch_size 512 \
    --mixed_precision \
    --epochs 50 \
    --patience 10 \
    --model_type deep \
    --experiment_name "optimized_full_run" \
    2>&1 | tee training_log.txt

echo "✅ Training completed at: $(date)"
echo "📄 Full log saved to: training_log.txt"
echo "📊 Results saved to: ./outputs/optimized_full_run/"