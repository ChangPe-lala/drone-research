"""
generate_result_figures.py
==========================

Generate publication/report figures from the EXISTING research outputs.

This script DOES NOT:
- retrain any model
- change CP1 / CP2 / CP3 decisions
- rerun Multi-Hub routing
- modify result CSV files

It only reads existing CSV/JSON outputs and saves PNG figures.

Run from project root:
    python generate_result_figures.py

Main output folders:
    output/figures/multiuav/validation/
    output/figures/multihub/
    output/figures/ablation/dji_only/   # already produced by run_full_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
RESULTS = OUTPUT / "results"
DATA = OUTPUT / "data"
FIGURES = OUTPUT / "figures"

VAL_FIG_DIR = FIGURES / "multiuav" / "validation"
MH_FIG_DIR = FIGURES / "multihub"

VAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
MH_FIG_DIR.mkdir(parents=True, exist_ok=True)

CASES = ["DJI_ONLY", "VTOL_ONLY", "COMBINED"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_csv(paths: list[Path], label: str) -> tuple[pd.DataFrame | None, Path | None]:
    path = _first_existing(paths)
    if path is None:
        print(f"[SKIP] {label}: source CSV not found")
        return None, None
    try:
        return pd.read_csv(path), path
    except Exception as exc:
        print(f"[SKIP] {label}: cannot read {path}: {type(exc).__name__}: {exc}")
        return None, path


def _save(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_path.relative_to(ROOT)}")


def _annotate_bars(ax, fmt="{:.3f}", rotation=0):
    for patch in ax.patches:
        value = patch.get_height()
        if np.isfinite(value):
            ax.annotate(
                fmt.format(value),
                (patch.get_x() + patch.get_width() / 2.0, value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                fontsize=8,
                rotation=rotation,
            )


def _short_case(case: str) -> str:
    return {
        "DJI_ONLY": "DJI",
        "VTOL_ONLY": "VTOL",
        "COMBINED": "Combined",
    }.get(str(case), str(case))


def _case_result_base(case: str) -> list[Path]:
    c = case.lower()
    return [
        RESULTS / "multiuav" / "validation" / c,
        RESULTS / "validation" / c,
    ]


def _cp1_paths(case: str) -> list[Path]:
    paths = []
    for base in _case_result_base(case):
        paths += [
            base / "cp1_feature_selection" / "feature_evidence.csv",
            base / "cp1_feature_screen" / "feature_evidence.csv",
        ]
    return paths


def _cp2_paths(case: str) -> list[Path]:
    paths = []
    for base in _case_result_base(case):
        paths += [
            base / "cp2_model_benchmark" / "model_evidence.csv",
            base / "cp2_model_selection" / "model_evidence.csv",
        ]
    return paths


def _finalized_paths(case: str, filename: str) -> list[Path]:
    paths = []
    for base in _case_result_base(case):
        paths.append(base / "finalized" / filename)
    return paths


def _validation_case_evidence_paths() -> list[Path]:
    return [
        RESULTS / "multiuav" / "validation" / "cp3_comparison" / "validation_case_evidence.csv",
        RESULTS / "multiuav" / "validation" / "validation_case_evidence.csv",
        RESULTS / "validation" / "cp3_comparison" / "validation_case_evidence.csv",
    ]


def _selected_config_paths() -> list[Path]:
    return [
        RESULTS / "multiuav" / "validation" / "selected_configuration.json",
        RESULTS / "validation" / "selected_configuration.json",
    ]


def _selected_config() -> dict:
    path = _first_existing(_selected_config_paths())
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# CP1 — feature-set comparison
# ---------------------------------------------------------------------------

def make_cp1_figures() -> list[Path]:
    frames = []
    used = []
    for case in CASES:
        df, path = _read_csv(_cp1_paths(case), f"CP1 {case}")
        if df is None or df.empty:
            continue
        if "Case" not in df.columns:
            df["Case"] = case
        frames.append(df)
        used.append(path)

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

    needed = {"Case", "Feature_Set", "CV_R2_mean_across_models", "CV_WMAPE_mean_across_models"}
    if not needed.issubset(df.columns):
        print(f"[SKIP] CP1 figures: missing columns {sorted(needed - set(df.columns))}")
        return []

    labels = [
        f"{_short_case(c)}\n{fs}"
        for c, fs in zip(df["Case"].astype(str), df["Feature_Set"].astype(str))
    ]
    x = np.arange(len(df))

    out = []

    plt.figure(figsize=(max(10, 1.15 * len(df)), 6))
    ax = plt.gca()
    ax.bar(x, pd.to_numeric(df["CV_R2_mean_across_models"], errors="coerce"))
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean CV R² across benchmark models")
    ax.set_title("CP1 Feature-Set Screening — CV R²")
    _annotate_bars(ax, "{:.3f}")
    p = VAL_FIG_DIR / "cp1_feature_set_cv_r2.png"
    _save(p)
    out.append(p)

    plt.figure(figsize=(max(10, 1.15 * len(df)), 6))
    ax = plt.gca()
    wmape_pct = pd.to_numeric(df["CV_WMAPE_mean_across_models"], errors="coerce") * 100.0
    ax.bar(x, wmape_pct)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean CV WMAPE (%)")
    ax.set_title("CP1 Feature-Set Screening — CV WMAPE")
    _annotate_bars(ax, "{:.2f}%")
    p = VAL_FIG_DIR / "cp1_feature_set_cv_wmape.png"
    _save(p)
    out.append(p)

    return out


# ---------------------------------------------------------------------------
# CP2 — one model benchmark figure per case
# ---------------------------------------------------------------------------

def make_cp2_figures() -> list[Path]:
    out = []
    selected = _selected_config()

    for case in CASES:
        df, _ = _read_csv(_cp2_paths(case), f"CP2 {case}")
        if df is None or df.empty:
            continue

        required = {"Model", "CV_R2_mean", "CV_WMAPE_mean"}
        if not required.issubset(df.columns):
            print(f"[SKIP] CP2 {case}: missing {sorted(required - set(df.columns))}")
            continue

        work = df.copy()
        work["CV_R2_mean"] = pd.to_numeric(work["CV_R2_mean"], errors="coerce")
        work["CV_WMAPE_mean"] = pd.to_numeric(work["CV_WMAPE_mean"], errors="coerce")
        work = work.sort_values("CV_R2_mean", ascending=False)

        title_suffix = ""
        if selected.get("selected_case") == case:
            title_suffix = f" — CP3 selected case ({selected.get('model', '')})"

        # R2
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        x = np.arange(len(work))
        ax.bar(x, work["CV_R2_mean"])
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(work["Model"].astype(str), rotation=40, ha="right")
        ax.set_ylabel("CV R²")
        ax.set_title(f"CP2 Model Benchmark — {_short_case(case)}{title_suffix}")
        _annotate_bars(ax, "{:.3f}")
        p = VAL_FIG_DIR / f"cp2_{case.lower()}_model_cv_r2.png"
        _save(p)
        out.append(p)

        # WMAPE
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        wmape = work["CV_WMAPE_mean"] * 100.0
        ax.bar(x, wmape)
        ax.set_xticks(x)
        ax.set_xticklabels(work["Model"].astype(str), rotation=40, ha="right")
        ax.set_ylabel("CV WMAPE (%)")
        ax.set_title(f"CP2 Model Benchmark — {_short_case(case)} WMAPE")
        _annotate_bars(ax, "{:.2f}%")
        p = VAL_FIG_DIR / f"cp2_{case.lower()}_model_cv_wmape.png"
        _save(p)
        out.append(p)

    return out


# ---------------------------------------------------------------------------
# Finalized holdout figures
# ---------------------------------------------------------------------------

def make_holdout_figures() -> list[Path]:
    out = []

    # DJI actual-vs-predicted is the final selected scientific model.
    dji, _ = _read_csv(
        _finalized_paths("DJI_ONLY", "holdout_predictions.csv"),
        "DJI holdout predictions",
    )
    if dji is not None:
        actual_col = "energy_efficiency"
        pred_col = "predicted_energy_efficiency"
        if {actual_col, pred_col}.issubset(dji.columns):
            actual = pd.to_numeric(dji[actual_col], errors="coerce")
            pred = pd.to_numeric(dji[pred_col], errors="coerce")
            mask = np.isfinite(actual) & np.isfinite(pred)

            plt.figure(figsize=(7, 7))
            ax = plt.gca()
            ax.scatter(actual[mask], pred[mask], alpha=0.75)
            low = float(min(actual[mask].min(), pred[mask].min()))
            high = float(max(actual[mask].max(), pred[mask].max()))
            ax.plot([low, high], [low, high], linewidth=1)
            ax.set_xlabel("Observed EE (m/Wh)")
            ax.set_ylabel("Predicted EE (m/Wh)")
            ax.set_title("DJI Untouched Holdout — Observed vs Predicted EE")
            p = VAL_FIG_DIR / "dji_holdout_actual_vs_predicted.png"
            _save(p)
            out.append(p)

    evidence, _ = _read_csv(
        _validation_case_evidence_paths(),
        "CP3 validation case evidence",
    )
    if evidence is not None and not evidence.empty:
        if {"Case", "R2"}.issubset(evidence.columns):
            work = evidence.copy()
            work["R2"] = pd.to_numeric(work["R2"], errors="coerce")
            x = np.arange(len(work))
            plt.figure(figsize=(8, 5.5))
            ax = plt.gca()
            ax.bar(x, work["R2"])
            ax.axhline(0.0, linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels([_short_case(x) for x in work["Case"]])
            ax.set_ylabel("Untouched holdout R²")
            ax.set_title("Validation Cases — Untouched Holdout R²")
            _annotate_bars(ax, "{:.3f}")
            p = VAL_FIG_DIR / "validation_case_holdout_r2.png"
            _save(p)
            out.append(p)

        if {"Case", "WMAPE"}.issubset(evidence.columns):
            work = evidence.copy()
            work["WMAPE_pct"] = pd.to_numeric(work["WMAPE"], errors="coerce") * 100
            x = np.arange(len(work))
            plt.figure(figsize=(8, 5.5))
            ax = plt.gca()
            ax.bar(x, work["WMAPE_pct"])
            ax.set_xticks(x)
            ax.set_xticklabels([_short_case(x) for x in work["Case"]])
            ax.set_ylabel("Untouched holdout WMAPE (%)")
            ax.set_title("Validation Cases — Untouched Holdout WMAPE")
            _annotate_bars(ax, "{:.2f}%")
            p = VAL_FIG_DIR / "validation_case_holdout_wmape.png"
            _save(p)
            out.append(p)

    # Diagnostic: Combined aggregate can hide weak within-platform generalization.
    comb, _ = _read_csv(
        _finalized_paths("COMBINED", "holdout_predictions.csv"),
        "Combined holdout predictions",
    )
    if comb is not None and {
        "dataset_id", "energy_efficiency", "predicted_energy_efficiency"
    }.issubset(comb.columns):
        rows = []
        for dataset_id, sub in comb.groupby("dataset_id"):
            y = pd.to_numeric(sub["energy_efficiency"], errors="coerce").to_numpy(float)
            p = pd.to_numeric(sub["predicted_energy_efficiency"], errors="coerce").to_numpy(float)
            valid = np.isfinite(y) & np.isfinite(p)
            y, p = y[valid], p[valid]
            if len(y) < 2:
                continue
            denom = np.sum((y - np.mean(y)) ** 2)
            r2 = np.nan if denom <= 0 else 1.0 - np.sum((y - p) ** 2) / denom
            rows.append({"Platform": str(dataset_id), "R2": r2, "N": len(y)})

        if rows:
            s = pd.DataFrame(rows)
            plt.figure(figsize=(7, 5))
            ax = plt.gca()
            x = np.arange(len(s))
            ax.bar(x, s["R2"])
            ax.axhline(0.0, linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(s["Platform"])
            ax.set_ylabel("Within-platform holdout R²")
            ax.set_title("Combined Model — Holdout R² by Platform")
            _annotate_bars(ax, "{:.3f}")
            p = VAL_FIG_DIR / "combined_holdout_r2_by_platform.png"
            _save(p)
            out.append(p)

    return out


# ---------------------------------------------------------------------------
# Segment Energy model benchmark
# ---------------------------------------------------------------------------

def make_segment_model_figures() -> list[Path]:
    df, _ = _read_csv(
        [
            RESULTS / "multihub" / "segment_model_checkpoint" / "segment_model_evidence.csv",
            RESULTS / "multihub" / "segment_model_evidence.csv",
        ],
        "G07 segment model evidence",
    )
    if df is None or df.empty:
        return []

    needed = {"Model", "CV_R2_mean", "CV_WMAPE_mean"}
    if not needed.issubset(df.columns):
        print(f"[SKIP] G07 figures: missing {sorted(needed - set(df.columns))}")
        return []

    work = df.copy()
    work["CV_R2_mean"] = pd.to_numeric(work["CV_R2_mean"], errors="coerce")
    work["CV_WMAPE_mean"] = pd.to_numeric(work["CV_WMAPE_mean"], errors="coerce")
    work = work.sort_values("CV_R2_mean", ascending=False)
    x = np.arange(len(work))
    out = []

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(x, work["CV_R2_mean"])
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(work["Model"].astype(str), rotation=40, ha="right")
    ax.set_ylabel("Flight-grouped CV R²")
    ax.set_title("G07 Segment Energy Model Benchmark — CV R²")
    _annotate_bars(ax, "{:.3f}")
    p = MH_FIG_DIR / "segment_model_cv_r2.png"
    _save(p)
    out.append(p)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(x, work["CV_WMAPE_mean"] * 100.0)
    ax.set_xticks(x)
    ax.set_xticklabels(work["Model"].astype(str), rotation=40, ha="right")
    ax.set_ylabel("Flight-grouped CV WMAPE (%)")
    ax.set_title("G07 Segment Energy Model Benchmark — CV WMAPE")
    _annotate_bars(ax, "{:.2f}%")
    p = MH_FIG_DIR / "segment_model_cv_wmape.png"
    _save(p)
    out.append(p)

    return out


# ---------------------------------------------------------------------------
# Multi-Hub topology
# ---------------------------------------------------------------------------

def make_multihub_topology() -> list[Path]:
    edges, _ = _read_csv(
        [DATA / "multihub" / "graph_edges.csv"],
        "Multi-Hub graph edges",
    )
    if edges is None or edges.empty:
        return []

    required = {
        "start_node", "end_node",
        "start_x", "start_y", "end_x", "end_y",
    }
    if not required.issubset(edges.columns):
        print(f"[SKIP] topology: missing {sorted(required - set(edges.columns))}")
        return []

    e = edges.copy()
    for col in ["start_x", "start_y", "end_x", "end_y"]:
        e[col] = pd.to_numeric(e[col], errors="coerce")
    e = e.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if e.empty:
        print("[SKIP] topology: no finite edge coordinates")
        return []

    plt.figure(figsize=(9, 8))
    ax = plt.gca()

    # Directed topology using the same coordinate space stored by G08.
    for _, row in e.iterrows():
        sx, sy = row["start_x"], row["start_y"]
        ex, ey = row["end_x"], row["end_y"]
        ax.plot([sx, ex], [sy, ey], linewidth=1, alpha=0.55)
        ax.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "alpha": 0.45},
        )

    node_pos = {}
    for _, row in e.iterrows():
        node_pos[str(row["start_node"])] = (row["start_x"], row["start_y"])
        node_pos[str(row["end_node"])] = (row["end_x"], row["end_y"])

    xs = [v[0] for v in node_pos.values()]
    ys = [v[1] for v in node_pos.values()]
    ax.scatter(xs, ys, s=35, zorder=3)

    prefix = "N_DJI_M100_"
    for node, (xv, yv) in node_pos.items():
        label = node[len(prefix):] if node.startswith(prefix) else node
        ax.annotate(label, (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Graph coordinate X")
    ax.set_ylabel("Graph coordinate Y")
    ax.set_title(f"Final DJI Multi-Hub Directed Topology ({len(node_pos)} nodes, {len(e)} edges)")
    ax.set_aspect("equal", adjustable="datalim")

    p = MH_FIG_DIR / "multihub_topology.png"
    _save(p)
    return [p]


# ---------------------------------------------------------------------------
# Final RQ5 figures
# ---------------------------------------------------------------------------

def _final_rq5_recommendations() -> tuple[pd.DataFrame | None, Path | None]:
    return _read_csv(
        [RESULTS / "multihub" / "final_rq5" / "mission_recommendations.csv"],
        "Final RQ5 mission recommendations",
    )


def make_final_rq5_figures() -> list[Path]:
    df, _ = _final_rq5_recommendations()
    if df is None or df.empty:
        return []

    out = []
    work = df.copy()

    if "mission_id" not in work.columns:
        work["mission_id"] = np.arange(1, len(work) + 1).astype(str)
    labels = work["mission_id"].astype(str).str.replace("DJI_M100_", "", regex=False)
    x = np.arange(len(work))

    # 1) Historical vs final recommended EE
    # Use observed_historical_ee if available, fallback to predicted_historical_ee
    hist_col = "observed_historical_ee" if "observed_historical_ee" in work.columns else "predicted_historical_ee"
    
    if {hist_col, "recommended_ee"}.issubset(work.columns):
        hist = pd.to_numeric(work[hist_col], errors="coerce")
        rec = pd.to_numeric(work["recommended_ee"], errors="coerce")

        plt.figure(figsize=(11, 6))
        ax = plt.gca()
        width = 0.38
        ax.bar(x - width / 2, hist, width=width, label="Historical")
        ax.bar(x + width / 2, rec, width=width, label="Recommended")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Mission ID (DJI_M100_ prefix omitted)")
        ax.set_ylabel("EE (m/Wh)")
        ax.set_title("Final RQ5 — Historical vs Recommended Energy Efficiency")
        ax.legend()
        p = MH_FIG_DIR / "historical_vs_recommended_ee.png"
        _save(p)
        out.append(p)

    # 2) Final policy improvement (two retained historical missions become 0%)
    if "predicted_ee_gain_pct" in work.columns:
        # If recommendation was historical, gain is 0. If hypothetical, it's predicted_ee_gain_pct
        imp = pd.Series(0.0, index=work.index)
        hyp_mask = work["recommended_option"] == "HYPOTHETICAL"
        imp[hyp_mask] = pd.to_numeric(work.loc[hyp_mask, "predicted_ee_gain_pct"], errors="coerce")
    elif "recommended_ee_improvement_pct" in work.columns:
        imp = pd.to_numeric(work["recommended_ee_improvement_pct"], errors="coerce")
    elif "ee_improvement_pct" in work.columns:
        imp = pd.to_numeric(work["ee_improvement_pct"], errors="coerce").clip(lower=0)
    else:
        imp = None

    if imp is not None:
        plt.figure(figsize=(11, 6))
        ax = plt.gca()
        ax.bar(x, imp)
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Mission ID (DJI_M100_ prefix omitted)")
        ax.set_ylabel("Recommended EE improvement (%)")
        ax.set_title("Final RQ5 — EE Improvement After Recommendation Policy")
        _annotate_bars(ax, "{:.2f}%")
        p = MH_FIG_DIR / "recommended_ee_improvement_pct.png"
        _save(p)
        out.append(p)

    # 3) Best hypothetical candidate energy-saving trade-off.
    es_col = "predicted_ee_gain_pct" if "predicted_ee_gain_pct" in work.columns else "energy_saving_pct"
    if es_col in work.columns:
        es = pd.to_numeric(work[es_col], errors="coerce")
        plt.figure(figsize=(11, 6))
        ax = plt.gca()
        ax.bar(x, es)
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Mission ID (DJI_M100_ prefix omitted)")
        ax.set_ylabel("Best-candidate energy saving (%)")
        ax.set_title("Final RQ5 — Absolute Energy Saving of Best Hypothetical Candidate")
        p = MH_FIG_DIR / "best_candidate_energy_saving_pct.png"
        _save(p)
        out.append(p)

    # 4) Detour ratio audit.
    dist_ratio_col = "predicted_candidate_distance" # need to calculate ratio
    hist_dist_col = "observed_historical_distance" if "observed_historical_distance" in work.columns else "predicted_historical_distance"
    
    if dist_ratio_col in work.columns and hist_dist_col in work.columns:
        ratio = pd.to_numeric(work[dist_ratio_col], errors="coerce") / pd.to_numeric(work[hist_dist_col], errors="coerce")
        
        plt.figure(figsize=(11, 6))
        ax = plt.gca()
        ax.bar(x, ratio)
        ax.axhline(1.0, linewidth=1, label="Historical distance")
        # Protocol cap is known from the final pipeline; use summary if available.
        cap = 1.5
        summary_path = RESULTS / "multihub" / "final_rq5" / "summary.csv"
        if summary_path.exists():
            try:
                s = pd.read_csv(summary_path)
                if "max_candidate_distance_ratio_vs_historical" in s.columns:
                    cap = float(s.iloc[0]["max_candidate_distance_ratio_vs_historical"])
            except Exception:
                pass
        ax.axhline(cap, linewidth=1, linestyle="--", label=f"Candidate cap = {cap:g}×")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Mission ID (DJI_M100_ prefix omitted)")
        ax.set_ylabel("Candidate distance / historical distance")
        ax.set_title("Final RQ5 — Candidate Route Distance-Ratio Audit")
        ax.legend()
        p = MH_FIG_DIR / "candidate_distance_ratio.png"
        _save(p)
        out.append(p)

    # 5) Final decision counts.
    decision_col = None
    if "recommended_option" in work.columns:
        decision_col = "recommended_option"
    elif "recommendation_decision" in work.columns:
        decision_col = "recommendation_decision"

    if decision_col:
        counts = work[decision_col].fillna("UNKNOWN").astype(str).value_counts()
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        xx = np.arange(len(counts))
        ax.bar(xx, counts.values)
        ax.set_xticks(xx)
        ax.set_xticklabels(counts.index, rotation=20, ha="right")
        ax.set_ylabel("Missions")
        ax.set_title("Final RQ5 — Recommendation Decisions")
        _annotate_bars(ax, "{:.0f}")
        p = MH_FIG_DIR / "recommendation_decision_counts.png"
        _save(p)
        out.append(p)

    # 6) Recommended hub frequency — only ACTUALLY recommended hypothetical routes.
    if "predicted_candidate_hubs" in work.columns:
        hub_work = work.copy()
        if "recommended_option" in hub_work.columns:
            hub_work = hub_work[hub_work["recommended_option"].astype(str) == "HYPOTHETICAL"]
        elif "recommendation_decision" in hub_work.columns:
            hub_work = hub_work[
                hub_work["recommendation_decision"].astype(str) == "USE_HYPOTHETICAL_ROUTE"
            ]
        hubs = hub_work["predicted_candidate_hubs"].replace("", np.nan).dropna().astype(str)
        if not hubs.empty:
            counts = hubs.value_counts()
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            xx = np.arange(len(counts))
            short_labels = [
                h.replace("N_DJI_M100_", "") for h in counts.index.astype(str)
            ]
            ax.bar(xx, counts.values)
            ax.set_xticks(xx)
            ax.set_xticklabels(short_labels, rotation=25, ha="right")
            ax.set_xlabel("Recommended hub node (prefix omitted)")
            ax.set_ylabel("Recommended missions")
            ax.set_title("Final RQ5 — Recommended Hub Frequency")
            _annotate_bars(ax, "{:.0f}")
            p = MH_FIG_DIR / "recommended_hub_frequency.png"
            _save(p)
            out.append(p)

    return out


# ---------------------------------------------------------------------------
# Figure index
# ---------------------------------------------------------------------------

def write_figure_index(paths: list[Path]) -> None:
    index_path = FIGURES / "RESULT_FIGURE_INDEX.txt"
    lines = [
        "RESULT FIGURE INDEX",
        "===================",
        "",
        "Generated by: python generate_result_figures.py",
        "",
        "Recommended MAIN REPORT figures:",
        "  1. multiuav/validation/cp1_feature_set_cv_r2.png",
        "     - CP1 feature-set evidence / RQ1-RQ2 context",
        "  2. multiuav/validation/cp2_dji_only_model_cv_r2.png",
        "     - RQ3 model comparison for final DJI setting",
        "  3. multiuav/validation/dji_holdout_actual_vs_predicted.png",
        "     - final DJI CatBoost generalization",
        "  4. finalized DJI SHAP figures (created by finalize_case.py)",
        "     - RQ1 / RQ4",
        "  5. ablation/dji_only/group_ablation_delta_r2.png",
        "     - RQ1 / RQ2 group contribution",
        "  6. ablation/dji_only/individual_ablation_delta_r2.png",
        "     - RQ1 / RQ2 feature contribution",
        "  7. multihub/segment_model_cv_r2.png",
        "     - G07 segment Energy model selection",
        "  8. multihub/multihub_topology.png",
        "     - final virtual-hub graph topology",
        "  9. multihub/historical_vs_recommended_ee.png",
        "     - RQ5 historical vs final recommended EE",
        " 10. multihub/recommended_ee_improvement_pct.png",
        "     - RQ5 final mission-level EE gains",
        "",
        "Supporting / appendix:",
        "  - CP1 WMAPE",
        "  - CP2 WMAPE and non-selected cases",
        "  - validation case R2 / WMAPE",
        "  - Combined within-platform R2 diagnostic",
        "  - segment model WMAPE",
        "  - best-candidate energy saving",
        "  - candidate distance-ratio audit",
        "  - recommendation decision counts",
        "  - recommended hub frequency",
        "",
        "Files generated in this run:",
    ]
    for path in paths:
        try:
            rel = path.relative_to(FIGURES)
        except Exception:
            rel = path
        lines.append(f"  - {rel}")
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {index_path.relative_to(ROOT)}")


def main() -> None:
    print("=" * 78)
    print("GENERATE RESEARCH RESULT FIGURES")
    print("=" * 78)
    print("No training / no feature selection / no routing rerun.")
    print()

    generated: list[Path] = []

    generated += make_cp1_figures()
    generated += make_cp2_figures()
    generated += make_holdout_figures()
    generated += make_segment_model_figures()
    generated += make_multihub_topology()
    generated += make_final_rq5_figures()

    write_figure_index(generated)

    print()
    print("=" * 78)
    print(f"COMPLETE — generated {len(generated)} figure(s)")
    print(f"Validation figures: {VAL_FIG_DIR}")
    print(f"Multi-Hub figures : {MH_FIG_DIR}")
    print(f"Figure index      : {FIGURES / 'RESULT_FIGURE_INDEX.txt'}")
    print("=" * 78)


if __name__ == "__main__":
    main()