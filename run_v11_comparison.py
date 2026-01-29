
import subprocess
import glob
import os
import shutil
from datetime import datetime

# Configuration
SCRIPT = "models/model_v11.py"
CONFIGS = {
    "korea": "configs/model_v11_korea.yaml",
    "singapore": "configs/model_v11_singapore.yaml"
}
OUTPUT_DIR = "simulation_run_v1/outputs/model_v11_comparison"
REPORT_FILE = os.path.join(OUTPUT_DIR, "v11_comparison_report.md")

def run_model(config_path):
    print(f"Running model with config: {config_path}")
    cmd = ["python", SCRIPT, "--config", config_path]
    result = subprocess.run(cmd, cwd="simulation_run_v1", text=True)
    if result.returncode != 0:
        print(f"Error running model for {config_path}")
        return None
    
    # Find the latest run folder
    # Assuming config defines experiment_name
    # We parse the yaml to get experiment name? Or just glob runs/?
    # Just glob runs/model_v11_*/*
    # Let's find the most recent directory in runs/
    
    # But wait, experiment_name varies.
    # Korea: model_v11_un_data
    # Singapore: model_v11_singapore
    
    # We can infer from metadata or just search
    if "korea" in config_path:
        exp_name = "model_v11_un_data"
    else:
        exp_name = "model_v11_singapore"
        
    runs_dir = os.path.join("simulation_run_v1/runs", exp_name)
    list_of_dirs = glob.glob(os.path.join(runs_dir, "*"))
    latest_dir = max(list_of_dirs, key=os.path.getctime)
    print(f"Latest run directory: {latest_dir}")
    return latest_dir

def copy_artifacts(run_dir, prefix):
    # Copy key images to OUTPUT_DIR with prefix
    # We define the TARGET name, and we search for it
    target_map = {
        "viv_forecast.png": "figures/viv/image_C_viv_pretty_*.png",
        "viv_candidates.png": "figures/viv/image_D_viv_candidates_vs_realized_*.png",
        "index_volume.png": "figures/index/index_volume_overlay.png",
        "tavi_demographic.png": "figures/index/tavi_demographic_contribution.png",
        "savr_demographic.png": "figures/index/savr_demographic_contribution.png",
        "projected_adoption.png": "figures/index/projected_adoption.png"
    }
    
    # Try to copy
    copied = []
    for dest_suffix, src_pattern in target_map.items():
        full_pattern = os.path.join(run_dir, src_pattern)
        candidates = glob.glob(full_pattern)
        
        if candidates:
            # Pick the first one (usually only one for the range)
            src_path = candidates[0]
            dest_name = f"{prefix}_{dest_suffix}"
            dest_path = os.path.join(OUTPUT_DIR, dest_name)
            shutil.copy2(src_path, dest_path)
            copied.append(dest_name)
        else:
            print(f"Warning: Could not find pattern {full_pattern}")
            
    return copied

def generate_report(korea_dir, singapore_dir):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    korea_imgs = copy_artifacts(korea_dir, "korea")
    sg_imgs = copy_artifacts(singapore_dir, "singapore")
    
    md = f"""# Model v11 Comparison Report: Korea vs Singapore
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report compares the projected ViV volume and Index Volume drivers for Korea and Singapore.

## 1. ViV Forecast (Image C)
**Korea** | **Singapore**
--- | ---
![Korea ViV](korea_viv_forecast.png) | ![Singapore ViV](singapore_viv_forecast.png)

## 2. ViV Candidates vs Realized (Image D)
**Korea** | **Singapore**
--- | ---
![Korea Candidates](korea_viv_candidates.png) | ![Singapore Candidates](singapore_viv_candidates.png)

## 3. Index Volume Projection
**Korea** | **Singapore**
--- | ---
![Korea Index](korea_index_volume.png) | ![Singapore Index](singapore_index_volume.png)

## 4. Demographic Impact on TAVI Volume
**Korea** | **Singapore**
--- | ---
![Korea Demographics](korea_tavi_demographic.png) | ![Singapore Demographics](singapore_tavi_demographic.png)

## 5. Demographic Impact on SAVR Volume
**Korea** | **Singapore**
--- | ---
![Korea Demographics](korea_savr_demographic.png) | ![Singapore Demographics](singapore_savr_demographic.png)

## 6. Projected Adoption
**Korea** | **Singapore**
--- | ---
![Korea Adoption](korea_projected_adoption.png) | ![Singapore Adoption](singapore_projected_adoption.png)

"""
    with open(REPORT_FILE, "w") as f:
        f.write(md)
    
    print(f"Report generated at {REPORT_FILE}")

def main():
    print("Starting Model v11 Comparison Run...")
    
    # Run Korea
    korea_dir = run_model(CONFIGS["korea"])
    if not korea_dir: return
    
    # Run Singapore
    singapore_dir = run_model(CONFIGS["singapore"])
    if not singapore_dir: return
    
    # Generate Report
    generate_report(korea_dir, singapore_dir)

if __name__ == "__main__":
    main()
