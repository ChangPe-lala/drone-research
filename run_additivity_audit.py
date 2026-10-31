import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.paths import OUTPUT_DIR

def run_additivity_audit():
    print("="*60)
    print("ADDITIVITY / SEGMENTATION-INVARIANCE AUDIT")
    print("="*60)
    
    features_path = OUTPUT_DIR / "data" / "multihub" / "edge_request_features.csv"
    model_path = OUTPUT_DIR / "models" / "multihub" / "segment_energy_model.pkl"
    sel_path = OUTPUT_DIR / "data" / "multihub" / "policy_selections.csv"
    
    df = pd.read_csv(features_path)
    model_data = joblib.load(model_path)
    sel_df = pd.read_csv(sel_path)
    
    pipeline = model_data["pipeline"]
    features = model_data["features"]
    
    # Split d into d1=0.4d, d2=0.6d to test asymmetry
    X_original = df[features].copy()
    f_d = pipeline.predict(X_original)
    
    X_40 = X_original.copy()
    X_40["segment_distance"] = X_original["segment_distance"] * 0.4
    f_d_40 = pipeline.predict(X_40)
    
    X_60 = X_original.copy()
    X_60["segment_distance"] = X_original["segment_distance"] * 0.6
    f_d_60 = pipeline.predict(X_60)
    
    f_d1_plus_d2 = f_d_40 + f_d_60
    
    df["predicted_energy_d"] = f_d
    df["predicted_energy_split"] = f_d1_plus_d2
    df["additivity_error_pct"] = (f_d1_plus_d2 - f_d) / f_d * 100
    df["abs_additivity_error_pct"] = df["additivity_error_pct"].abs()
    
    # Get requests where P1 != P2
    disagree_reqs = set()
    for _, row in sel_df.iterrows():
        if pd.notna(row["P1_SHORTEST_PATH_ID"]) and pd.notna(row["P2_MIN_ENERGY_PATH_ID"]):
            if row["P1_SHORTEST_PATH_ID"] != row["P2_MIN_ENERGY_PATH_ID"]:
                disagree_reqs.add(row["request_id"])
                
    df["is_disagreement_req"] = df["request_id"].isin(disagree_reqs)
    
    # Merge UAV type from route_requests
    req_df = pd.read_csv(OUTPUT_DIR / "data" / "multihub" / "route_requests.csv")
    df = df.merge(req_df[["request_id", "uav_type"]], on="request_id", how="left")
    
    def report_stats(subset_df, name):
        if len(subset_df) == 0:
            return
        med = subset_df["abs_additivity_error_pct"].median()
        p90 = np.percentile(subset_df["abs_additivity_error_pct"], 90)
        p95 = np.percentile(subset_df["abs_additivity_error_pct"], 95)
        mean_err = subset_df["additivity_error_pct"].mean()
        print(f"--- {name} (N={len(subset_df)}) ---")
        print(f"  Mean Directional Additivity Error: {mean_err:.2f}%")
        print(f"  Median Abs Additivity Error: {med:.2f}%")
        print(f"  P90 Abs Additivity Error: {p90:.2f}%")
        print(f"  P95 Abs Additivity Error: {p95:.2f}%")
        print()
        
    report_stats(df, "ALL EDGES (Global)")
    
    dji_df = df[df["uav_type"] == "Quadrotor"]
    vtol_df = df[df["uav_type"] == "Fixed-wing"]
    
    report_stats(dji_df, "Quadrotor (DJI)")
    report_stats(vtol_df, "Fixed-wing (VTOL)")
    
    disagree_df = df[df["is_disagreement_req"]]
    report_stats(disagree_df, "Requests with P1 != P2 (Shortest vs Min-Energy Disagreement)")

if __name__ == "__main__":
    run_additivity_audit()
