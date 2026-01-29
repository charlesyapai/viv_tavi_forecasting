#!/usr/bin/env python3
import argparse
import subprocess
import shutil
import sys
import yaml
from datetime import datetime as dt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

def run_command(cmd, cwd=None):
    log.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Command failed with code {result.returncode}")
        log.error(f"STDOUT:\n{result.stdout}")
        log.error(f"STDERR:\n{result.stderr}")
        sys.exit(result.returncode)
    else:
        # log.info(f"STDOUT:\n{result.stdout}")
        pass
    return result

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline for TAVI/SAVR Forecasting")
    parser.add_argument("--country", choices=["singapore", "korea"], default="singapore")
    parser.add_argument("--start-year", type=int, default=2012, help="Start year for index data (inclusive)")
    parser.add_argument("--config-template", default="simulation_run_v1/configs/model_v12_singapore.yaml")
    parser.add_argument("--sigmoid-mode", default="free", help="Sigmoid mode: free, fixed_cap, fixed_midpoint")
    parser.add_argument("--fixed-midpoint-val", type=float, default=2013, help="Value for x0 if using fixed_midpoint mode")
    args = parser.parse_args()

    # 1. Setup Directories
    base_dir = Path("simulation_run_v1")
    ts = dt.now().strftime("%Y-%m-%d-%H%M%S")
    run_name = f"model_v12_{args.country}"
    run_tag = f"{ts}_{args.country}_unified_pipeline"
    
    # Final Output Directory: runs/model_v12_singapore/YYYY-MM-DD-HHMMSS_singapore_unified_pipeline
    out_root = base_dir / "runs" / run_name / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Pipeline initialized. Output Directory: {out_root}")

    # Sub-directories
    index_out_dir = out_root / "index_projections"
    index_out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Run Index Projection Model
    log.info("--- Step 1: Generating Index Projections ---")
    
    if args.country == "singapore":
        model_script = "models/final_adoption_model_singapore.py"
    else:
        # Fallback or specific script for Korea if unified later, 
        # currently assuming user wants this mainly for Singapore as per request.
        model_script = "models/final_adoption_model.py" 

    # For now, final_adoption_model.py (Korea) might not have --start-year argument yet
    # But user specifically asked for Singapore pipeline.
    
    cmd_index = [
        "python", model_script,
        "--out-dir", str(index_out_dir.absolute()),
        "--start-year", str(args.start_year)
    ]
    
    if args.country == "korea":
        cmd_index.extend([
            "--sigmoid-mode", str(args.sigmoid_mode),
            "--fixed-midpoint-val", str(args.fixed_midpoint_val)
        ])
    
    run_command(cmd_index, cwd=str(base_dir))
    log.info(f"Index projections saved to {index_out_dir}")

    # Verify Output
    expected_report = index_out_dir / "final_calculation_report.csv"
    if not expected_report.exists():
        log.error("Index projection failed to produce final_calculation_report.csv")
        sys.exit(1)

    # 3. Create Dynamic Config
    log.info("--- Step 2: Configuring Simulation Parameters ---")
    
    # Read Template
    with open(args.config_template, 'r') as f:
        config = yaml.safe_load(f)

    # Update Index Projection Paths (to point to the just-generated files)
    # We point both TAVI and SAVR to the new report
    config['index_projection']['tavi']['method'] = 'external'
    config['index_projection']['tavi']['external_file'] = str(expected_report.absolute())
    config['index_projection']['tavi']['external_col'] = 'pred_TAVI'

    config['index_projection']['savr']['method'] = 'external'
    config['index_projection']['savr']['external_file'] = str(expected_report.absolute())
    config['index_projection']['savr']['external_col'] = 'pred_SAVR'

    # Save Temporary Config used for this run
    run_config_path = out_root / "pipeline_config.yaml"
    with open(run_config_path, 'w') as f:
        yaml.dump(config, f)
    
    log.info(f"Configuration updated and saved to {run_config_path}")

    # 4. Run Monte Carlo Simulation
    log.info("--- Step 3: Running Monte Carlo Simulation ---")
    
    cmd_sim = [
        "python", "models/model_v12.py",
        "--config", str(run_config_path.absolute()),
        "--override-out-dir", str(out_root.absolute()),
        "--log-level", "INFO"
    ]

    run_command(cmd_sim, cwd=str(base_dir))
    log.info("Simulation completed successfully.")

    # 5. Final Asset Organization (Optional: Zip everything)
    log.info("--- Step 4: Finalizing Outputs ---")
    zip_name = f"{args.country}_unified_results_{ts}.zip"
    zip_path = base_dir / "runs" / zip_name
    
    # Create valid zip command
    # cd to the parent of out_root to zip the folder cleanly? 
    # Or just zip the full path. Let's zip relative to simulation_run_v1/runs
    
    # Construct relative path for zipping
    rel_path_to_run = Path("runs") / run_name / run_tag
    
    cmd_zip = [
        "zip", "-r", str(zip_path.absolute()),
        str(rel_path_to_run)
    ]
    
    run_command(cmd_zip, cwd=str(base_dir))
    log.info(f"All results archived to: {zip_path}")
    log.info("Pipeline Finished.")

if __name__ == "__main__":
    main()
