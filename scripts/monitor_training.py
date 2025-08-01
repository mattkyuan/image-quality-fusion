#!/usr/bin/env python3
"""
Training progress monitor
"""
import time
import os
from pathlib import Path

def monitor_training():
    """Monitor training progress by checking output files"""
    print("🔍 Monitoring training progress...")
    
    # Check for key indicator files (relative to project root)
    indicators = [
        "../training_data/features.csv",
        "../training_data/embeddings.npy", 
        "../outputs/optimized_full_run/config.json",
        "../outputs/optimized_full_run/model_best.pth",
        "../outputs/optimized_full_run/training_results.json"
    ]
    
    last_sizes = {}
    
    while True:
        print(f"\n📊 Status check at {time.strftime('%H:%M:%S')}")
        
        for file_path in indicators:
            path = Path(file_path)
            if path.exists():
                size_mb = path.stat().st_size / (1024*1024)
                
                # Check if file is growing (indicating active work)
                growth = ""
                if file_path in last_sizes:
                    if size_mb > last_sizes[file_path]:
                        growth = f" (+{size_mb - last_sizes[file_path]:.1f}MB)"
                
                print(f"✅ {path.name}: {size_mb:.1f}MB{growth}")
                last_sizes[file_path] = size_mb
            else:
                print(f"⏳ {Path(file_path).name}: Not created yet")
        
        # Check for log file
        if Path("../training_log.txt").exists():
            with open("../training_log.txt", 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        print(f"📝 Latest: {last_line}")
        
        # Check if training is complete
        if Path("../outputs/optimized_full_run/training_results.json").exists():
            print("🎉 Training completed successfully!")
            break
            
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")