import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import OUTPUT_DIR

def compute_ee_gain():
    res_path = OUTPUT_DIR / "results" / "multihub" / "policy_comparison.csv"
    if not res_path.exists():
        print("policy_comparison.csv not found!")
        return
        
    df = pd.read_csv(res_path)
    
    # Calculate EE = Distance / Energy (m/Wh)
    # P1 (Shortest Distance)
    df["EE_P1"] = df["shortest_distance"] / df["shortest_energy"]
    
    # P2 (Min Energy)
    df["EE_P2"] = df["min_energy_distance"] / df["min_energy_energy"]
    
    # EE Gain % = (EE_P2 - EE_P1) / EE_P1 * 100
    df["EE_Gain_pct"] = (df["EE_P2"] - df["EE_P1"]) / df["EE_P1"] * 100
    
    # P0 (Original)
    # Check if there is original distance
    paths_path = OUTPUT_DIR / "data" / "multihub" / "candidate_paths.csv"
    sel_path = OUTPUT_DIR / "data" / "multihub" / "policy_selections.csv"
    paths_df = pd.read_csv(paths_path).set_index("path_id")
    sel_df = pd.read_csv(sel_path)
    
    # We can join with sel_df to get P0_ORIGINAL_PATH_ID and get its distance
    df = df.merge(sel_df[["request_id", "P0_ORIGINAL_PATH_ID"]], on="request_id", how="left")
    
    p0_ee_list = []
    for _, row in df.iterrows():
        p0_id = row["P0_ORIGINAL_PATH_ID"]
        if pd.notna(p0_id):
            p0_path = paths_df.loc[p0_id]
            ee0 = p0_path["total_distance"] / p0_path["predicted_total_energy"]
            p0_ee_list.append(ee0)
        else:
            p0_ee_list.append(None)
            
    df["EE_P0"] = p0_ee_list
    df["EE_Gain_vs_P0_pct"] = (df["EE_P2"] - df["EE_P0"]) / df["EE_P0"] * 100
    
    print("="*60)
    print("ENERGY EFFICIENCY (EE) GAIN %")
    print("="*60)
    print(f"Total Evaluated Requests: {len(df)}")
    
    # Mean EE Gain vs Shortest
    mean_ee_gain_p1 = df["EE_Gain_pct"].mean()
    max_ee_gain_p1 = df["EE_Gain_pct"].max()
    print(f"\n1. P2 (Min-Energy) vs P1 (Shortest Distance):")
    print(f"   - Average EE Gain: {mean_ee_gain_p1:.2f}%")
    print(f"   - Max EE Gain: {max_ee_gain_p1:.2f}%")
    
    # Focus only on the 39 disagreements
    disagree_df = df[df["shortest_distance"] != df["min_energy_distance"]]
    if not disagree_df.empty:
        print(f"\n   Among {len(disagree_df)} Disagreements (Routable ODs):")
        print(f"   - Average EE Gain: {disagree_df['EE_Gain_pct'].mean():.2f}%")
        
    # EE Gain vs Original
    with_p0 = df.dropna(subset=["EE_Gain_vs_P0_pct"])
    if not with_p0.empty:
        mean_ee_gain_p0 = with_p0["EE_Gain_vs_P0_pct"].mean()
        max_ee_gain_p0 = with_p0["EE_Gain_vs_P0_pct"].max()
        print(f"\n2. P2 (Min-Energy) vs P0 (Original Historical Path):")
        print(f"   - Evaluated on {len(with_p0)} flights.")
        print(f"   - Average EE Gain: {mean_ee_gain_p0:.2f}%")
        print(f"   - Max EE Gain: {max_ee_gain_p0:.2f}%")
        
    out_dir = OUTPUT_DIR / "results" / "multihub"
    df.to_csv(out_dir / "ee_gain_results.csv", index=False)
    print(f"\nSaved detailed EE Gain to {out_dir / 'ee_gain_results.csv'}")
    
if __name__ == "__main__":
    compute_ee_gain()
