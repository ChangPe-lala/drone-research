import json
import pandas as pd
from pathlib import Path

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def main():
    rs_dir = Path('.')
    
    print_section("Flight-Level Holdout Metrics")
    for case, name in [('dji_only', 'DJI'), ('vtol_only', 'VTOL'), ('combined', 'Combined')]:
        manifest_file = rs_dir / f'output/results/multiuav/validation/{case}/finalized/selection_manifest.json'
        if manifest_file.exists():
            with open(manifest_file) as f:
                data = json.load(f)
                h_metrics = data.get('holdout_metrics', {})
                print(f"{name} Holdout R2: {h_metrics.get('R2', 'N/A')}")
                print(f"{name} Holdout WMAPE: {h_metrics.get('WMAPE', 'N/A')}")
                
    print_section("Segment Energy Metrics")
    seg_file = rs_dir / 'output/results/multihub/segment_model_evidence.csv'
    if seg_file.exists():
        df = pd.read_csv(seg_file)
        # Look for XGBoost
        row = df[df['Model'] == 'XGBoost'].iloc[0]
        print(f"Segment XGBoost CV R2: {row['CV_R2_mean']} +/- {row['CV_R2_std']}")
        print(f"Segment XGBoost CV WMAPE: {row['CV_WMAPE_mean']} +/- {row['CV_WMAPE_std']}")
        
    print_section("SHAP Values")
    shap_file = rs_dir / 'output/results/multiuav/validation/dji_only/finalized/shap/shap_importance.csv'
    if shap_file.exists():
        print(pd.read_csv(shap_file))
        
    print_section("Final RQ5 Metrics (Route Recommendations)")
    summary_file = rs_dir / 'output/results/multihub/final_rq5/summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            print(json.dumps(data, indent=2))
            
    print_section("Route Baseline Summary")
    route_summary = rs_dir / 'output/results/multihub/route_baseline_summary.csv'
    if route_summary.exists():
        df = pd.read_csv(route_summary)
        print(df[['strategy', 'mean_distance', 'mean_predicted_energy', 'mean_predicted_ee', 'mean_ee_gain_vs_historical_pct', 'mean_energy_saving_vs_historical_pct']])
        
    print_section("Topology Audit - N5")
    audit_file = rs_dir / 'output/results/multihub/topology_audit/dominant_hub_removed_policy_selections.csv'
    if audit_file.exists():
        df = pd.read_csv(audit_file)
        print("Topology Audit columns:", df.columns.tolist())
        # Let's just print a couple rows if any
        if not df.empty:
            print(df.head(1))
            
    # Also read summary of topology audit
    audit_summary_file = rs_dir / 'output/results/multihub/topology_audit/summary.csv'
    if audit_summary_file.exists():
        print(pd.read_csv(audit_summary_file))
        
    print_section("Sensitivity Summaries")
    sensitivity_dir = rs_dir / 'output/sensitivity_runs'
    if sensitivity_dir.exists():
        for d in sensitivity_dir.iterdir():
            if d.is_dir():
                # check summary
                s_file = d / 'results/multihub/topology_audit/dominant_hub_removed_policy_selections.csv'
                if s_file.exists():
                    df = pd.read_csv(s_file)
                    print(f"Sensitivity {d.name} rows: {len(df)}")
                    
    print_section("Error Analysis DJI Holdout")
    err_file = rs_dir / 'output/results/multiuav/validation/dji_only/finalized/holdout_predictions.csv'
    if err_file.exists():
        df = pd.read_csv(err_file)
        errors = (df['energy_efficiency'] - df['predicted_energy_efficiency']).abs()
        print(f"MAE: {errors.mean():.4f}")
        print(f"Median AE: {errors.median():.4f}")
        print(f"RMSE: {(errors**2).mean()**0.5:.4f}")
        print(f"Max residual: {errors.max():.4f}")
        print(f"<= 1: {(errors <= 1).mean():.4f}")
        print(f"<= 2: {(errors <= 2).mean():.4f}")

if __name__ == '__main__':
    main()
