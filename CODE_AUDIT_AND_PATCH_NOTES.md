# CODE AUDIT & PATCH NOTES — Multi-UAV → Multi-Hub

## Scope
This patch audits and corrects the current Multi-UAV EE pipeline and the planned-wind integration before D09–D13 / Multi-Hub final experiments.

## Critical fixes applied

1. **Primary target corrected**
   - `PRIMARY_TARGET = "energy_efficiency"`
   - `SECONDARY_TARGET = "battery_consumed_wh"`

2. **Distance/Energy temporal support made consistent**
   - Both distance and energy now use adjacent segments satisfying:
     - finite endpoints
     - `0 < dt <= 30 s`
   - Energy uses trapezoidal integration on real timestamps.
   - Legacy no-gap-filter energy remains available only through the audit script using `max_gap_sec=None`.

3. **Battery current sign handling corrected**
   - No unconditional `abs(current)`.
   - Sign normalization is used only when negative current is the dominant convention.
   - Small fractions of negative current remain signed for sensitivity/audit.

4. **Planned wind function implemented centrally**
   - `compute_planned_wind_features()` added to `target_calculation.py`.
   - Meteorological `wind_dir_deg` is interpreted as wind **FROM**.
   - Converted to wind **TO** using:
     - `wind_to = (wind_from + 180) % 360`
   - Relative-angle convention:
     - `0° = tailwind`
     - `90° = crosswind`
     - `180° = headwind`
   - Headwind component convention:
     - positive = headwind
     - negative = tailwind
     - `H = -W cos(theta_rel)`
   - Crosswind magnitude:
     - `C = |W sin(theta_rel)|`

5. **Geographic route direction corrected**
   - Geographic coordinates use great-circle initial bearing.
   - Local XY uses `atan2(dx, dy)` for compass-style heading.
   - Removed an `np.isclose` issue that incorrectly classified meter-scale geographic displacements as identical.

6. **Observed vs planned relative wind separated**
   - `relative_wind_angle` remains a legacy observed/post-flight analysis variable.
   - `planned_relative_wind_angle`, `planned_headwind_component`, and `planned_crosswind_component` are separate route-weather variables.

7. **Important empirical route-proxy audit (DJI)**
   Running the patched DJI D04 on the included cleaned dataset produced:
   - 194 flights
   - mean actual path distance: ~598.25 m
   - mean Start→End displacement: ~2.91 m
   - mean endpoint/path ratio: ~0.004935 = **0.4935%**
   - 94.3% of flights have endpoint/path ratio < 1%
   - 100% have endpoint/path ratio < 5%

   Therefore, historical flight endpoints are a poor proxy for planned route direction for DJI. The three planned-wind columns are still generated for diagnostics, but feature sets F1/F2/F3 are **guarded** and cannot be treated as final pre-flight inputs unless true planned route geometry is available. Diagnostic use requires an explicit flag.

8. **Aerodynamic feature ablation formalized**
   - `F0_BASE`: 11 clean common pre-flight features
   - `F1_ANGLE`: F0 + planned relative wind angle
   - `F2_COMPONENTS`: F0 + planned headwind/crosswind components
   - `F3_ALL_AERO`: F0 + angle + both components
   - F1/F2/F3 require true planned geometry for final use.

9. **D07/D08 preprocessing leakage fixed**
   - Imputation/scaling is fitted inside training folds/source domain only.
   - No `X.fillna(X.mean())` using the full CV dataset before splitting.
   - D07/D08 now support explicit feature-set selection.

10. **Target integration audit added**
    - `src/analysis/target_integration_audit.py`
    - Outputs legacy vs gap-filtered energy, signed/positive-only/absolute-current variants, EE variants, gap statistics and VTOL configuration stability.

11. **Multi-Hub planned wind moved to the correct place**
    - G08 graph edges now carry start/end node coordinates.
    - G10 computes planned wind features for each **edge × request** using edge geometry + request forecast weather.
    - No route bearing is required as a model feature.
    - G07 segment training uses the same selected feature schema as G10.

12. **Segment target sign/gap handling corrected**
    - Segment energy no longer hardcodes `normalize_sign=True`.
    - Uses dominant sign convention and the shared 30 s gap rule.

13. **D13 hard-coded final model removed**
    - It no longer automatically declares XGBoost Universal as final.
    - D13 builds an evidence table first.
    - Final freezing requires explicit approval in `decisions/final_model_decision.yaml`.

14. **Dataset extension runner repaired/extended**
    - Includes D00–D13 plus D06A target audit and D07A aerodynamic ablation.
    - `--from-stage` now takes precedence correctly.

## Validation performed in this patched copy

- Python compile check: PASS
- Unit tests for target + wind physics: **7/7 PASS**
- DJI D04 full feature engineering: PASS
  - output: 194 flights × 27 columns
- DJI target gap audit:
  - flights with gap >30 s: 0/194
  - legacy vs corrected energy change: 0%
  - Therefore the 30 s gap bug does **not** affect DJI in the included cleaned dataset.
- VTOL target audit was not executable inside this ZIP because `data/processed/vtol_cleaned_flight_data.csv` is not included. Run D02 locally first / use your existing VTOL cleaned file, then run D06A.

## Recommended next run order on the machine containing the full VTOL processed data

```powershell
python run_dataset_extension.py --stage D06A
```

Review `output/results/multiuav/target_audit_summary.csv` and VTOL per-flight audit outputs.

Then regenerate final target/features:

```powershell
python run_dataset_extension.py --stage D03
python run_dataset_extension.py --stage D04
python run_dataset_extension.py --stage D05
python run_dataset_extension.py --stage D06
```

Run clean baseline:

```powershell
python src/modeling/01_within_domain.py --feature-set F0_BASE
python src/modeling/02_cross_domain.py --feature-set F0_BASE
```

Do **not** treat F1/F2/F3 as final flight-level pre-flight features until true planned mission geometry is available. For diagnostic-only endpoint-proxy ablation:

```powershell
python src/modeling/01c_aero_feature_ablation.py --allow-endpoint-proxy --include-cross-domain
```

Then proceed:

```powershell
python run_dataset_extension.py --stage D09
python run_dataset_extension.py --stage D10
python run_dataset_extension.py --stage D11
python run_dataset_extension.py --stage D12
python run_dataset_extension.py --stage D13
```

D13 will create/require `decisions/final_model_decision.yaml` before freezing a final artifact.
