import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.paths import OUTPUT_DIR


def _safe_pct(num, den):
    return (num / den * 100.0) if den else 0.0


def _describe(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def run_sanity_audit():
    print("=" * 70)
    print("FINAL G14 SANITY AUDIT")
    print("=" * 70)

    data_dir = OUTPUT_DIR / "data" / "multihub"
    results_dir = OUTPUT_DIR / "results" / "multihub"
    results_dir.mkdir(parents=True, exist_ok=True)

    sel_path = data_dir / "policy_selections.csv"
    paths_path = data_dir / "candidate_paths.csv"
    requests_path = data_dir / "route_requests.csv"

    for p in [sel_path, paths_path, requests_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    sel_df = pd.read_csv(sel_path)
    paths_raw = pd.read_csv(paths_path)
    requests_df = pd.read_csv(requests_path)

    if "is_routable" not in requests_df.columns:
        raise ValueError(
            "route_requests.csv has no 'is_routable' column. "
            "Run the fixed G09 first."
        )

    # Use only paths that are valid for routing/model evaluation.
    valid_paths = paths_raw[
        (paths_raw["path_energy_is_valid"] == True)
        & (paths_raw["path_platform_match"] == True)
    ].copy()

    paths_df = paths_raw.set_index("path_id")
    valid_paths_idx = valid_paths.set_index("path_id")

    req_meta = requests_df.set_index("request_id")
    routable_lookup = req_meta["is_routable"].astype(bool).to_dict()

    # ------------------------------------------------------------------
    # 1. Disagreement reconciliation
    # ------------------------------------------------------------------
    total_evaluated = 0
    all_disagreements = 0
    eligible_count = 0
    eligible_disagreements = 0

    disagreement_rows = []

    for _, row in sel_df.iterrows():
        req_id = row["request_id"]
        p1_id = row["P1_SHORTEST_PATH_ID"]
        p2_id = row["P2_MIN_ENERGY_PATH_ID"]

        if pd.isna(p1_id) or pd.isna(p2_id):
            continue
        if p1_id not in paths_df.index or p2_id not in paths_df.index:
            continue

        total_evaluated += 1
        is_eligible = bool(routable_lookup.get(req_id, False))
        disagree = p1_id != p2_id

        if disagree:
            all_disagreements += 1

        if is_eligible:
            eligible_count += 1
            if disagree:
                eligible_disagreements += 1

        if disagree:
            p1 = paths_df.loc[p1_id]
            p2 = paths_df.loc[p2_id]

            disagreement_rows.append({
                "request_id": req_id,
                "is_routable": is_eligible,
                "p1_path_id": p1_id,
                "p2_path_id": p2_id,
                "p1_distance": float(p1["total_distance"]),
                "p2_distance": float(p2["total_distance"]),
                "p1_energy": float(p1["predicted_total_energy"]),
                "p2_energy": float(p2["predicted_total_energy"]),
                "p1_num_hops": float(p1["num_hops"]),
                "p2_num_hops": float(p2["num_hops"]),
            })

    print("\n--- 1. Disagreement Rate Reconciliation ---")
    print(f"Total evaluated requests: {total_evaluated}")
    print(f"Eligible Multi-Hub requests (G09 is_routable=True): {eligible_count}")
    print(f"All-request disagreements: {all_disagreements}")
    print(f"Eligible-request disagreements: {eligible_disagreements}")
    print(
        f"Disagreement rate (all requests): "
        f"{_safe_pct(all_disagreements, total_evaluated):.2f}%"
    )
    print(
        f"Disagreement rate (eligible requests): "
        f"{_safe_pct(eligible_disagreements, eligible_count):.2f}%"
    )

    # ------------------------------------------------------------------
    # 2. Unique routable OD count — use G09 flag, not candidate-row count
    # ------------------------------------------------------------------
    routable_req_df = requests_df[requests_df["is_routable"].astype(bool)].copy()
    unique_ods = routable_req_df[
        ["source_node", "destination_node"]
    ].drop_duplicates()

    print("\n--- 2. Unique Routable OD Pair Count ---")
    print(f"Routable requests: {len(routable_req_df)}")
    print(f"Unique routable OD pairs: {len(unique_ods)}")

    # ------------------------------------------------------------------
    # 3. Segmentation-invariance diagnostics
    # ------------------------------------------------------------------
    print("\n--- 3. Segmentation-Invariance Diagnostics ---")

    valid_paths["wh_per_m"] = (
        valid_paths["predicted_total_energy"]
        / valid_paths["total_distance"].replace(0, np.nan)
    )
    valid_paths["mean_edge_length"] = (
        valid_paths["total_distance"]
        / valid_paths["num_hops"].replace(0, np.nan)
    )

    corr_hops = valid_paths["wh_per_m"].corr(valid_paths["num_hops"])
    corr_len = valid_paths["wh_per_m"].corr(valid_paths["mean_edge_length"])

    print(f"Valid/platform-matched candidate paths: {len(valid_paths)}")
    print(f"Wh/m vs Num Hops correlation: {corr_hops:.4f}")
    print(f"Wh/m vs Mean Edge Length correlation: {corr_len:.4f}")

    disagree_df = pd.DataFrame(disagreement_rows)
    eligible_disagree_df = disagree_df[
        disagree_df["is_routable"] == True
    ].copy() if not disagree_df.empty else pd.DataFrame()

    if not eligible_disagree_df.empty:
        eligible_disagree_df["p1_wh_per_m"] = (
            eligible_disagree_df["p1_energy"]
            / eligible_disagree_df["p1_distance"].replace(0, np.nan)
        )
        eligible_disagree_df["p2_wh_per_m"] = (
            eligible_disagree_df["p2_energy"]
            / eligible_disagree_df["p2_distance"].replace(0, np.nan)
        )
        eligible_disagree_df["p1_mean_edge_length"] = (
            eligible_disagree_df["p1_distance"]
            / eligible_disagree_df["p1_num_hops"].replace(0, np.nan)
        )
        eligible_disagree_df["p2_mean_edge_length"] = (
            eligible_disagree_df["p2_distance"]
            / eligible_disagree_df["p2_num_hops"].replace(0, np.nan)
        )

        print(
            f"\nFor {len(eligible_disagree_df)} eligible P1-vs-P2 disagreements:"
        )
        print(
            "  P1 Shortest: "
            f"Avg Hops={eligible_disagree_df['p1_num_hops'].mean():.2f}, "
            f"Avg Mean-Edge={eligible_disagree_df['p1_mean_edge_length'].mean():.2f}m, "
            f"Wh/m={eligible_disagree_df['p1_wh_per_m'].mean():.6f}"
        )
        print(
            "  P2 MinEnergy: "
            f"Avg Hops={eligible_disagree_df['p2_num_hops'].mean():.2f}, "
            f"Avg Mean-Edge={eligible_disagree_df['p2_mean_edge_length'].mean():.2f}m, "
            f"Wh/m={eligible_disagree_df['p2_wh_per_m'].mean():.6f}"
        )

    # ------------------------------------------------------------------
    # 4. P0 historical-baseline backtracking/loop analysis
    # ------------------------------------------------------------------
    print("\n--- 4. P0 Historical Path Backtracking / Loop Analysis ---")

    p0_valid_df = sel_df.dropna(subset=["P0_ORIGINAL_PATH_ID"]).copy()
    print(f"P0 requests with recovered historical baseline: {len(p0_valid_df)}")

    backtrack_stats = []

    for _, row in p0_valid_df.iterrows():
        req_id = row["request_id"]
        p0_id = row["P0_ORIGINAL_PATH_ID"]
        p1_id = row["P1_SHORTEST_PATH_ID"]
        p2_id = row["P2_MIN_ENERGY_PATH_ID"]

        if (
            p0_id not in paths_df.index
            or p1_id not in paths_df.index
            or p2_id not in paths_df.index
        ):
            continue

        path0 = paths_df.loc[p0_id]
        path1 = paths_df.loc[p1_id]
        path2 = paths_df.loc[p2_id]

        nodes_0 = [
            x.strip()
            for x in str(path0["node_sequence"]).split(",")
            if x.strip()
        ]
        has_repeated_nodes = len(nodes_0) != len(set(nodes_0))

        dist0 = float(path0["total_distance"])
        dist1 = float(path1["total_distance"])
        dist2 = float(path2["total_distance"])

        e0 = float(path0["predicted_total_energy"])
        e1 = float(path1["predicted_total_energy"])
        e2 = float(path2["predicted_total_energy"])

        saving_vs_original_pct = (
            (e0 - e2) / e0 * 100.0
            if e0 > 0
            else np.nan
        )

        dataset_id = (
            req_meta.loc[req_id, "dataset_id"]
            if req_id in req_meta.index
            else "UNKNOWN"
        )
        uav_type = (
            req_meta.loc[req_id, "uav_type"]
            if req_id in req_meta.index
            else "UNKNOWN"
        )
        is_routable = bool(routable_lookup.get(req_id, False))

        backtrack_stats.append({
            "request_id": req_id,
            "dataset_id": dataset_id,
            "uav_type": uav_type,
            "is_routable": is_routable,
            "p0_dist": dist0,
            "p1_dist": dist1,
            "p2_dist": dist2,
            "p0_energy": e0,
            "p1_energy": e1,
            "p2_energy": e2,
            "p0_hops": float(path0["num_hops"]),
            "p1_hops": float(path1["num_hops"]),
            "p2_hops": float(path2["num_hops"]),
            "p0_has_repeated_nodes": has_repeated_nodes,
            "p0_to_p1_dist_ratio": (
                dist0 / dist1 if dist1 > 0 else np.nan
            ),
            "p0_to_p2_dist_ratio": (
                dist0 / dist2 if dist2 > 0 else np.nan
            ),
            "p0_to_p2_energy_ratio": (
                e0 / e2 if e2 > 0 else np.nan
            ),
            "model_implied_saving_pct": saving_vs_original_pct,
        })

    bk_df = pd.DataFrame(backtrack_stats)

    if bk_df.empty:
        print("No valid P0 rows available for audit.")
    else:
        loops = int(bk_df["p0_has_repeated_nodes"].sum())
        no_loops = len(bk_df) - loops

        print(
            f"P0 paths with repeated nodes: "
            f"{loops}/{len(bk_df)} "
            f"({_safe_pct(loops, len(bk_df)):.2f}%)"
        )
        print(f"P0 paths without repeated nodes: {no_loops}/{len(bk_df)}")

        print(f"Average P0 distance: {bk_df['p0_dist'].mean():.2f} m")
        print(f"Average P1 distance: {bk_df['p1_dist'].mean():.2f} m")
        print(f"Average P2 distance: {bk_df['p2_dist'].mean():.2f} m")
        print(
            f"Average P0/P1 distance ratio: "
            f"{bk_df['p0_to_p1_dist_ratio'].mean():.2f}x"
        )
        print(
            f"Median P0/P1 distance ratio: "
            f"{bk_df['p0_to_p1_dist_ratio'].median():.2f}x"
        )

        s = _describe(bk_df["model_implied_saving_pct"])
        print("\nModel-implied saving vs predicted original path:")
        print(f"  Mean   : {s['mean']:.2f}%")
        print(f"  Median : {s['median']:.2f}%")
        print(f"  P75    : {s['p75']:.2f}%")
        print(f"  P90    : {s['p90']:.2f}%")
        print(f"  Max    : {s['max']:.2f}%")

        high50 = int((bk_df["model_implied_saving_pct"] > 50).sum())
        high80 = int((bk_df["model_implied_saving_pct"] > 80).sum())
        print(
            f"Requests with >50% model-implied saving: "
            f"{high50}/{len(bk_df)}"
        )
        print(
            f"Requests with >80% model-implied saving: "
            f"{high80}/{len(bk_df)}"
        )

        # Compare loop vs non-loop groups.
        for label, subset in [
            ("P0 WITH repeated nodes", bk_df[bk_df["p0_has_repeated_nodes"] == True]),
            ("P0 WITHOUT repeated nodes", bk_df[bk_df["p0_has_repeated_nodes"] == False]),
        ]:
            print(f"\n{label}: n={len(subset)}")
            if subset.empty:
                continue
            ds = _describe(subset["model_implied_saving_pct"])
            print(
                f"  Saving mean/median: "
                f"{ds['mean']:.2f}% / {ds['median']:.2f}%"
            )
            print(
                f"  Avg P0/P1 distance ratio: "
                f"{subset['p0_to_p1_dist_ratio'].mean():.2f}x"
            )
            print(
                f"  Median P0/P1 distance ratio: "
                f"{subset['p0_to_p1_dist_ratio'].median():.2f}x"
            )

        # By dataset/platform.
        print("\nBy dataset:")
        for dataset_id, subset in bk_df.groupby("dataset_id"):
            ds = _describe(subset["model_implied_saving_pct"])
            loop_rate = _safe_pct(
                int(subset["p0_has_repeated_nodes"].sum()),
                len(subset),
            )
            print(
                f"  {dataset_id}: n={len(subset)}, "
                f"saving mean={ds['mean']:.2f}%, "
                f"median={ds['median']:.2f}%, "
                f"loop_rate={loop_rate:.2f}%, "
                f"avg P0/P1 ratio={subset['p0_to_p1_dist_ratio'].mean():.2f}x"
            )

        # Only the 30 G09-routable requests.
        eligible_bk = bk_df[bk_df["is_routable"] == True].copy()
        print(
            f"\nP0 audit restricted to G09-routable requests: "
            f"n={len(eligible_bk)}"
        )
        if not eligible_bk.empty:
            ds = _describe(
                eligible_bk["model_implied_saving_pct"]
            )
            print(
                f"  Saving mean/median: "
                f"{ds['mean']:.2f}% / {ds['median']:.2f}%"
            )
            print(
                f"  Avg P0/P1 distance ratio: "
                f"{eligible_bk['p0_to_p1_dist_ratio'].mean():.2f}x"
            )

        # Correlation: if close to +1, very large P0 saving is largely explained
        # by P0 being much longer than the shortest path.
        corr_ratio_saving = bk_df[
            "p0_to_p1_dist_ratio"
        ].corr(
            bk_df["model_implied_saving_pct"]
        )
        print(
            f"\nCorrelation(P0/P1 distance ratio, saving vs P0): "
            f"{corr_ratio_saving:.4f}"
        )

        # Save detailed P0 audit for report/debugging.
        audit_path = results_dir / "G14_p0_sanity_audit.csv"
        bk_df.to_csv(audit_path, index=False)
        print(f"Saved detailed P0 audit to: {audit_path}")

    # Save disagreement audit too.
    if not disagree_df.empty:
        disagreement_path = results_dir / "G14_disagreement_audit.csv"
        disagree_df.to_csv(disagreement_path, index=False)
        print(f"Saved disagreement audit to: {disagreement_path}")

    print("\n" + "=" * 70)
    print("G14 SANITY AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_sanity_audit()