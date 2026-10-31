# Kiểm tra code: Table III – cách tổng hợp các metric

## 1. Kết luận

Table III được tính đúng theo thiết kế hiện tại. Mỗi route được tính distance, predicted energy và predicted energy efficiency (EE) ở cấp mission. Sau đó, Table III lấy trung bình riêng từng metric trên 15 missions.

Với mission (i):

\[
EE_i=\frac{D_i}{\hat E_i},
\]

trong đó:

- (D_i): tổng khoảng cách route;
- \(\hat E_i\): tổng predicted segment energy của route;
- (EE_i): predicted distance per Wh của route.

Table III báo:

\[
\overline D=\frac{1}{N}\sum_{i=1}^{N}D_i,
\]

\[
\overline{\hat E}=\frac{1}{N}\sum_{i=1}^{N}\hat E_i,
\]

\[
\overline{EE}=\frac{1}{N}\sum_{i=1}^{N}\frac{D_i}{\hat E_i}.
\]

Do đó, nhìn chung:

\[
\overline{EE}\ne\frac{\overline D}{\overline{\hat E}}.
\]

Ví dụ với Historical:

```text
Mean distance                = 598.850900 m
Mean predicted energy        = 21.294492 Wh
Mean mission-level EE        = 28.714217 m/Wh
Mean distance / mean energy  = 28.122338 m/Wh
```

Sự khác biệt này là hệ quả của cách tổng hợp các tỷ lệ mission-level, không phải lỗi công thức.

## 2. Cách tính route distance

Khoảng cách của một route bằng tổng khoảng cách các directed edges thuộc route đó.

Nguồn: [routing_graph.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/routing_graph.py:88)

```python
def path_distance(graph: nx.DiGraph, node_path: list[str]) -> float:
    return float(
        sum(
            graph[node_path[i]][node_path[i + 1]]["segment_distance"]
            for i in range(len(node_path) - 1)
        )
    )
```

Tương ứng:

\[
D_i=\sum_{e\in r_i}D_e.
\]

## 3. Cách tính predicted route energy

Predicted route energy bằng tổng predicted segment energy của các edges thuộc route.

Nguồn: [enumerate_candidate_paths.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/enumerate_candidate_paths.py:132)

```python
energies: list[float] = []
all_energy_valid = True

for edge_id in edge_ids:
    value = energy_lookup.get((request_id, edge_id))
    is_valid = bool(
        valid_lookup.get((request_id, edge_id), False)
    )

    if value is None or pd.isna(value) or not is_valid:
        all_energy_valid = False
        break

    energies.append(float(value))

total_energy = (
    float(sum(energies))
    if all_energy_valid
    else float("nan")
)
```

Tương ứng:

\[
\hat E_i=\sum_{e\in r_i}\hat E_e.
\]

Historical P0 và các policy P1–P4 đều được chấm điểm bằng cùng segment model, cùng feature space và cùng cách cộng predicted segment energy. Đây là model-to-model comparison chính của routing experiment.

## 4. Cách tính predicted EE

Predicted EE được tính riêng cho mỗi route sau khi đã tính distance và predicted total energy.

Nguồn: [enumerate_candidate_paths.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/enumerate_candidate_paths.py:144)

```python
predicted_ee = (
    float(distance) / total_energy
    if all_energy_valid and total_energy > 0
    else float("nan")
)
```

Các giá trị route-level được lưu bằng code:

```python
records.append(
    {
        "total_distance": float(distance),
        "predicted_total_energy": total_energy,
        "predicted_ee": predicted_ee,
    }
)
```

Tương ứng:

\[
EE_i=\frac{D_i}{\hat E_i}.
\]

## 5. Hàm percentage change dùng chung

Nguồn: [strategy_comparison.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/strategy_comparison.py:39)

```python
def _pct_change(new_value: float, baseline_value: float) -> float:
    if not np.isfinite(baseline_value) or baseline_value == 0:
        return float("nan")
    return float(
        (new_value - baseline_value)
        / baseline_value
        * 100.0
    )
```

Hàm này được dùng cho distance change và EE gain. Historical là baseline trong paired comparison.

## 6. Distance change so với Historical

Nguồn: [strategy_comparison.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/strategy_comparison.py:141)

```python
row[f"{prefix}_distance_change_vs_historical_pct"] = _pct_change(
    float(path["total_distance"]),
    historical_distance,
)
```

Tương ứng:

\[
\Delta D_i=\frac{D_{strategy,i}-D_{historical,i}}
{D_{historical,i}}\times100.
\]

Quy ước dấu:

- \(\Delta D_i>0\): strategy dài hơn Historical;
- \(\Delta D_i<0\): strategy ngắn hơn Historical.

## 7. Energy saving so với Historical

Code định nghĩa energy saving dương là giảm predicted energy.

Nguồn: [strategy_comparison.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/strategy_comparison.py:45)

```python
def _energy_saving(
    strategy_energy: float,
    baseline_energy: float,
) -> float:
    """Positive values mean that the strategy consumes less Energy."""
    return -_pct_change(strategy_energy, baseline_energy)
```

Áp dụng cho từng mission:

```python
row[energy_saving_column] = _energy_saving(
    float(path["predicted_total_energy"]),
    historical_energy,
)
```

Tương ứng:

\[
EnergySaving_i=
\frac{\hat E_{historical,i}-\hat E_{strategy,i}}
{\hat E_{historical,i}}\times100.
\]

Quy ước dấu:

- \(EnergySaving_i>0\): strategy sử dụng ít predicted Wh hơn Historical;
- \(EnergySaving_i<0\): strategy sử dụng nhiều predicted Wh hơn Historical.

Nếu Table III sử dụng ký hiệu \(\Delta E\), cần định nghĩa rõ rằng đây là **energy saving**, với giá trị dương biểu thị giảm năng lượng. Để tránh reviewer hiểu nhầm, tên cột rõ ràng nhất là `Energy saving (%)`.

## 8. EE gain so với Historical

Nguồn: [strategy_comparison.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/strategy_comparison.py:154)

```python
row[ee_gain_column] = _pct_change(
    float(path["predicted_ee"]),
    historical_ee,
)
```

Tương ứng:

\[
EEGain_i=
\frac{EE_{strategy,i}-EE_{historical,i}}
{EE_{historical,i}}\times100.
\]

Quy ước dấu:

- \(EEGain_i>0\): strategy có predicted distance-per-Wh cao hơn Historical;
- \(EEGain_i<0\): strategy có predicted distance-per-Wh thấp hơn Historical.

## 9. Cách tạo các mean trong Table III

Nguồn: [strategy_comparison.py](C:/Users/Admin/Downloads/R/RS_updated_same_structure/RS/src/multihub/strategy_comparison.py:180)

```python
row = {
    "strategy": label,
    "objective": objective,
    "n_missions": int(len(comparison)),
    "mean_distance": float(
        comparison[f"{prefix}_distance"].mean()
    ),
    "mean_predicted_energy": float(
        comparison[f"{prefix}_predicted_energy"].mean()
    ),
    "mean_predicted_ee": float(
        comparison[f"{prefix}_predicted_ee"].mean()
    ),
}
```

Các paired percentage changes được tính cho từng mission trước, sau đó mới lấy mean:

```python
row.update(
    {
        "mean_distance_change_vs_historical_pct": float(
            comparison[
                f"{prefix}_distance_change_vs_historical_pct"
            ].mean()
        ),
        "mean_energy_saving_vs_historical_pct": float(
            comparison[
                "min_energy_saving_vs_historical_pct"
                if prefix == "min_energy"
                else f"{prefix}_energy_saving_vs_historical_pct"
            ].mean()
        ),
        "mean_ee_gain_vs_historical_pct": float(
            comparison[
                "max_ee_gain_vs_historical_pct"
                if prefix == "max_ee"
                else f"{prefix}_ee_gain_vs_historical_pct"
            ].mean()
        ),
    }
)
```

Do đó:

\[
\overline{\Delta D}
=\frac{1}{N}\sum_{i=1}^{N}\Delta D_i,
\]

\[
\overline{EnergySaving}
=\frac{1}{N}\sum_{i=1}^{N}EnergySaving_i,
\]

\[
\overline{EEGain}
=\frac{1}{N}\sum_{i=1}^{N}EEGain_i.
\]

## 10. Các giá trị Table III đã được tái kiểm tra

| Strategy | Mean distance (m) | Mean predicted energy (Wh) | Mean predicted EE (m/Wh) | Energy saving (%) | EE gain (%) |
|---|---:|---:|---:|---:|---:|
| Historical | 598.851 | 21.294 | 28.714 | 0.000 | 0.000 |
| Shortest-distance | 590.860 | 21.254 | 28.376 | +0.196 | -1.054 |
| Minimum-energy | 590.860 | 21.254 | 28.376 | +0.196 | -1.054 |
| Maximum-EE | 645.981 | 22.195 | 29.729 | -4.234 | +3.545 |
| Multi-objective | 627.575 | 21.695 | 29.528 | -1.893 | +2.874 |

Các giá trị trên là mean của 15 paired historical missions.

## 11. Câu trả lời hoàn chỉnh cho reviewer

> Route distance is calculated as the sum of edge distances, while predicted route energy is calculated as the sum of segment-energy predictions along the route. For each mission and strategy, predicted EE is then calculated as \(EE_i=D_i/\hat E_i\). Table III reports separate arithmetic means of mission-level distance, predicted energy, and EE. Therefore, the reported mean EE is \(\frac{1}{N}\sum_i D_i/\hat E_i\), not \(\overline D/\overline{\hat E}\). Distance change, energy saving, and EE gain are first calculated as paired mission-level percentage changes relative to the model-scored historical route and are subsequently averaged across the 15 missions. Positive energy-saving values denote lower predicted energy consumption, whereas negative values denote increased predicted energy consumption.

## 12. Note ngắn đề xuất đặt dưới Table III

> Distance, predicted energy, and EE are arithmetic means of mission-level values. Mean EE is mean(\(D_i/\hat E_i\)), not mean(\(D_i\))/mean(\(\hat E_i\)). Percentage changes are means of paired mission-level changes relative to the model-scored historical route. Positive energy-saving values denote reduced predicted energy consumption.
