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
    # Get project root from script location  
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # scripts/training/ -> project_root
    
    indicators = [
        project_root / "training_data/features.csv",
        project_root / "training_data/embeddings.npy", 
        project_root / "outputs/fixed_run/config.json",
        project_root / "outputs/fixed_run/model_best.pth",
        project_root / "outputs/fixed_run/training_results.json"
    ]
    
    last_sizes = {}
    
    while True:
        print(f"\n📊 Status check at {time.strftime('%H:%M:%S')}")
        
        for file_path in indicators:
            path = file_path
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
        log_file = project_root / "training_log.txt"
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        print(f"📝 Latest: {last_line}")
        
        # Check if training is complete
        if (project_root / "outputs/fixed_run/training_results.json").exists():
            print("🎉 Training completed successfully!")
            break
            
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")