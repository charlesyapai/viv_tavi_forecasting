import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(run_dir: Path):
    """Load TAVI and SAVR projection files."""
    index_dir = run_dir / "tables" / "index"
    tavi_path = index_dir / "tavi_with_projection.csv"
    savr_path = index_dir / "savr_with_projection.csv"

    if not tavi_path.exists() or not savr_path.exists():
        raise FileNotFoundError(f"Could not find projection files in {index_dir}")

    tavi = pd.read_csv(tavi_path)
    savr = pd.read_csv(savr_path)
    
    tavi["procedure"] = "TAVI"
    savr["procedure"] = "SAVR"
    
    df = pd.concat([tavi, savr], ignore_index=True)
    df["plot_src"] = df["src"].apply(lambda x: "observed" if x == "observed" else "projected")
    return df

def plot_historical(df: pd.DataFrame, out_dir: Path):
    """Plot side-by-side comparison of observed data."""
    # Filter for observed data only
    obs = df[df["plot_src"] == "observed"].copy()
    
    # Aggregate by year and procedure
    agg = obs.groupby(["year", "procedure"])["count"].sum().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg, x="year", y="count", hue="procedure", palette=["#1f77b4", "#ff7f0e"])
    
    plt.title("Historical Adoption: TAVI vs SAVR")
    plt.xlabel("Year")
    plt.ylabel("Total Procedures")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Procedure")
    
    out_path = out_dir / "historical_adoption.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved historical plot to {out_path}")
    plt.close()

def plot_projection(df: pd.DataFrame, out_dir: Path):
    """Plot timeline showing historical and projected trends."""
    # Aggregate by year, procedure, and source
    agg = df.groupby(["year", "procedure", "plot_src"])["count"].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # Plot lines
    sns.lineplot(data=agg, x="year", y="count", hue="procedure", style="plot_src", 
                 markers=True, dashes={"observed": "", "projected": (2, 2)},
                 palette=["#1f77b4", "#ff7f0e"])
    
    # Add a vertical line separating observed/projected if possible
    # Assuming the transition happens at the last observed year
    last_obs_year = agg[agg["plot_src"] == "observed"]["year"].max()
    if pd.notna(last_obs_year):
        plt.axvline(x=last_obs_year + 0.5, color="gray", linestyle="--", alpha=0.5, label="Projection Start")

    plt.title("Projected Adoption Trends: TAVI vs SAVR")
    plt.xlabel("Year")
    plt.ylabel("Total Procedures")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Procedure / Type")
    
    out_path = out_dir / "projected_adoption.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved projection plot to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze TAVI vs SAVR adoption trends.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the simulation run directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots (defaults to run_dir/analysis)")
    
    args = parser.parse_args()
    
    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
        
    # Setup output directory
    out_dir = args.out_dir if args.out_dir else args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.run_dir}...")
    df = load_data(args.run_dir)
    
    print("Generating plots...")
    plot_historical(df, out_dir)
    plot_projection(df, out_dir)
    
    print("Done.")

if __name__ == "__main__":
    main()
