# IMPLEMENTATION PLAN — MULTI-HUB GRAPH ROUTING FOR RS2
## Không hard-code dataset-specific logic

---

## 1. Mục tiêu

Xây dựng module Multi-Hub trên dữ liệu đã được preprocessing và cleaning hoàn tất.

Core flow:

```text
CLEAN / NORMALIZED TRAJECTORY DATA
        ↓
Detect / Define Hub Nodes
        ↓
Segment Historical Trajectories
        ↓
Build Segment-Level Dataset
        ↓
Train Segment Energy-First Model
        ↓
Build Directed Graph
        ↓
Create Valid Routing Requests
        ↓
Enumerate Valid Paths
        ↓
Predict Energy for Each Edge
        ↓
Aggregate Energy + EE for Each Path
        ↓
Compare Routing Policies
        ↓
Recommend Route
        ↓
Optional:
OOD + Uncertainty + Risk-Aware Routing
```

Plan này chỉ mô tả implementation core.

Không hard-code:
- tên route cụ thể
- flight ID cụ thể
- tên hub cụ thể
- số lượng hub
- số lượng path
- rule cleaning của từng dataset
- dataset nào chắc chắn dùng để dựng graph
- topology chưa được kiểm chứng
- threshold chưa được audit
- kết quả mong muốn

---

# 2. Nguyên tắc kiến trúc

Multi-Hub module **không tự clean raw data**.

Upstream pipeline chịu trách nhiệm:

```text
RAW DATA
    ↓
EXTRACTION
    ↓
CLEANING
    ↓
FEATURE / TARGET GENERATION
    ↓
NORMALIZED DATA
```

Multi-Hub chỉ bắt đầu từ:

```text
NORMALIZED DATA
```

Nếu dataset sau này thay đổi:
- số flight
- số route
- feature optional
- topology
- số node
- số path

thì core Multi-Hub không phải viết lại.

Chỉ có:
- adapter
- config
- normalized input
- runtime graph

thay đổi.

---

# 3. Input contract

Multi-Hub module yêu cầu tối thiểu:

```text
flight
time

position_x
position_y
position_z

speed
altitude
payload
```

và các environmental features mà model cuối thực sự sử dụng.

Ví dụ nếu final feature schema có:

```text
temperature_c
dew_point_c
humidity_pct
wind_speed_ms
wind_gust_ms
wind_dir_deg
precipitation_mm
pressure_hpa
relative_wind_angle
```

thì Multi-Hub chỉ reuse các cột đó.

Nếu có:

```text
date
route
dataset_id
uav_type
physics-derived features
```

thì giữ như metadata / model input nếu cần.

Ground truth dùng cho train/evaluation:

```text
battery_consumed_wh
energy_efficiency
```

Không thêm feature hình học mới nếu final methodology không cần.

---

# 4. Dataset adapter layer

Nếu có nhiều dataset:

```text
src/multihub/adapters/
├── base_adapter.py
├── dataset_a_adapter.py
└── dataset_b_adapter.py
```

Base interface:

```python
load_data()
get_flights()
get_trajectory(flight_id)
get_operational_features(flight_id)
get_environment_features(flight_id)
get_optional_route_metadata(flight_id)
```

Core Multi-Hub không được chứa:

```python
if route == ...
if flight in ...
if dataset == ...
```

Các rule riêng của dataset nằm ở:
- preprocessing
- cleaning
- adapter

---

# 5. Config

Tạo:

```text
config/multihub_config.py
```

Nên có:

```python
GRAPH_DIRECTED = True

HUB_PLACEMENT_METHOD = ...
HUB_DISTANCE_THRESHOLD_M = ...
MIN_HUB_SUPPORT = ...

MIN_SEGMENT_DISTANCE_M = ...

MAX_PATH_HOPS = ...
MAX_PATH_DISTANCE_RATIO = ...

USE_EXISTING_RELATIVE_WIND_ANGLE = True

RANDOM_STATE = 42
```

Không hard-code threshold trực tiếp trong algorithm.

---

# 6. G00 — Load normalized data

File:

```text
src/multihub/load_multihub_data.py
```

Input:
- cleaned / normalized dataset output

Validate:

```text
[ ] flight tồn tại
[ ] time hợp lệ
[ ] position tồn tại
[ ] model-required features tồn tại
[ ] không có flight rỗng
[ ] coordinate type được xác định
```

Output:
- normalized trajectory dataframe

Không thực hiện dataset-specific cleaning ở đây.

---

# 7. G01 — Build trajectory index

File:

```text
src/multihub/build_trajectory_index.py
```

Mỗi flight tạo metadata:

```text
flight
dataset_id
uav_type

start_position
end_position

n_points
total_distance

date
route metadata nếu có
```

Distance:
- reuse existing RS2 distance implementation nếu phù hợp.

Output:

```text
output/data/multihub/trajectory_index.csv
```

---

# 8. G02 — Detect / define hub nodes

File:

```text
src/multihub/detect_hub_nodes.py
```

Hub là node nằm:
- trên observed trajectory
- hoặc rất gần các trajectories

Cho phép nhiều method.

## Method A — Route intersection

Hub candidate tại nơi:
- 2 hoặc nhiều trajectories giao nhau.

## Method B — Near intersection

Nếu trajectories không cắt chính xác do GPS noise:

```text
distance(point_or_segment_A, point_or_segment_B)
    <= HUB_DISTANCE_THRESHOLD_M
```

→ candidate hub.

Distance:
- reuse existing spatial distance utility nếu phù hợp.

## Method C — On-route hub placement

Đặt hub trực tiếp trên observed trajectory.

Có thể theo:
- cumulative distance interval
- cumulative distance percentage
- configured positions

Các tỷ lệ / khoảng cách phải nằm trong config.

Không hard-code trong code.

---

# 9. G03 — Cluster hub candidates

File:

```text
src/multihub/cluster_hub_nodes.py
```

Mục tiêu:
- nhiều candidate points rất gần nhau → một graph node.

Có thể dùng:

```text
DBSCAN
```

Input:
- candidate coordinates

Output:

```text
output/data/multihub/graph_nodes.csv
```

Columns:

```text
node_id
node_type

position_x
position_y
position_z

source_method
support_count
```

`node_type`:

```text
START
DESTINATION
HUB
INTERMEDIATE
```

Nếu muốn node nằm đúng trên observed trajectory:
- ưu tiên medoid thay centroid.

---

# 10. G04 — Map trajectory to ordered nodes

File:

```text
src/multihub/map_trajectory_nodes.py
```

Cho mỗi flight:
- xác định thứ tự node mà trajectory đi qua.

Ví dụ generic:

```text
Node_0
  ↓
Node_1
  ↓
Node_2
  ↓
Node_3
```

Output:

```text
flight
ordered_node_sequence
```

File:

```text
output/data/multihub/trajectory_node_mapping.csv
```

---

# 11. G05 — Segment trajectories

File:

```text
src/multihub/segment_trajectories.py
```

Mỗi pair node liên tiếp tạo một segment.

Ví dụ:

```text
Trajectory:
Node_0 → Node_1 → Node_2 → Node_3
```

Segments:

```text
S1 = Node_0 → Node_1
S2 = Node_1 → Node_2
S3 = Node_2 → Node_3
```

Mỗi segment lưu:

```text
segment_id
source_flight

start_node
end_node

start_time
end_time
```

Giữ telemetry rows của segment để feature engineering.

---

# 12. G06 — Build segment-level dataset

File:

```text
src/multihub/build_segment_features.py
```

Output:

```text
output/data/multihub/segment_level_features.csv
```

Mỗi row = 1 historical segment.

Tối thiểu:

```text
segment_id
source_flight

dataset_id
uav_type

start_node
end_node

segment_distance
segment_duration
segment_energy_wh

speed
altitude
payload

environment features final model sử dụng

relative_wind_angle nếu upstream đã tạo đúng

date / route metadata nếu cần grouping
```

Các công thức hiện đã có trong RS2:

```text
energy integration
distance
duration
EE base definition
```

→ **REUSE EXISTING RS2 IMPLEMENTATION**

Chỉ áp dụng trên timestamp interval của segment.

---

# 13. Segment-level model là dependency bắt buộc

Nếu current model hiện tại là:

```text
1 whole flight → 1 energy value
```

thì không được tự động dùng model đó để predict:

```text
1 segment → 1 energy value
```

Nếu Multi-Hub ghép nhiều segments thành một path:

```text
Segment 1
+
Segment 2
+
Segment 3
=
Path
```

thì phải có segment-level energy model.

Default implementation:

```text
segment features
    ↓
Segment Energy-First Model
    ↓
segment_energy_wh
```

---

# 14. G07 — Train Segment Energy-First Model

File:

```text
src/multihub/train_segment_energy_model.py
```

Target:

```text
segment_energy_wh
```

Input:
- các feature biết trước / route-known / environmental features.

Có thể gồm:

```text
speed
altitude
payload
segment_distance

environmental features

relative_wind_angle
physics-derived features
```

nhưng chỉ khi các feature này nằm trong final methodology.

Không dùng:

```text
segment_energy_wh
actual post-flight target
actual segment duration
```

nếu final use case là pre-flight route recommendation.

`segment_distance` được phép dùng nếu:
- graph edge length đã biết trước route decision.

---

# 15. Không dùng `bearing`

Plan mới **bỏ hoàn toàn `bearing`**.

Không tạo:

```text
edge_bearing_deg
route_bearing
bearing feature
bearing utility
```

Không dùng bearing cho:
- graph
- model
- path scoring

Nếu `relative_wind_angle` đã tồn tại và được upstream tạo đúng:
- reuse trực tiếp.

Nếu một scenario mới cần recompute `relative_wind_angle` nhưng hiện tại không có đủ thông tin để recompute một cách nhất quán:
- không làm trong core Multi-Hub.
- xử lý ở upstream feature pipeline trước khi đưa vào Multi-Hub.

Core Multi-Hub không chịu trách nhiệm tái tạo wind-route geometry.

---

# 16. Relative wind angle

Nếu final model dùng:

```text
relative_wind_angle
```

thì Multi-Hub chỉ nhận giá trị đã chuẩn hóa từ upstream.

Không tự tính lại trong core.

Nếu upstream không cung cấp:
- không thêm `relative_wind_angle` vào segment model.

Feature schema phải thống nhất trước khi train segment model.

---

# 17. G08 — Build directed graph

File:

```text
src/multihub/build_route_graph.py
```

Khuyến nghị:

```python
networkx.DiGraph
```

Node:
- graph nodes

Edge:
- valid historical/configured segments

Mỗi edge lưu:

```text
edge_id
start_node
end_node

segment_distance

historical_support_count
supporting_segment_ids
```

Graph topology **không lưu một predicted energy cố định**.

Lý do:
- energy phụ thuộc request condition.

---

# 18. Edge support

Một edge được thêm khi:

```text
[ ] segment hợp lệ
[ ] start_node hợp lệ
[ ] end_node hợp lệ
[ ] có distance
[ ] đủ data để build model feature
[ ] support rule trong config được thỏa
```

Nếu nhiều historical segments có cùng:

```text
start_node → end_node
```

graph topology chỉ cần một edge.

Nhưng giữ:

```text
supporting_segment_ids
```

để:
- train
- validate
- audit support.

---

# 19. G09 — Build routing requests

File:

```text
src/multihub/build_route_requests.py
```

Một routing request:

```text
request_id
source_node
destination_node
```

Operational condition:

```text
speed
altitude
payload
```

Environmental condition:
- đúng các feature final model cần.

Có thể thêm:

```text
dataset_id
uav_type
date
```

Không hard-code source/destination name.

---

# 20. Edge geometry và request condition phải tách riêng

## Edge giữ:

```text
start_node
end_node
segment_distance
connectivity
```

## Request giữ:

```text
uav_type
speed
altitude
payload
environment
```

Scoring:

```text
EDGE
+
REQUEST
    ↓
feature vector
    ↓
segment energy model
    ↓
predicted edge energy
```

---

# 21. G10 — Build edge-request features

File:

```text
src/multihub/build_edge_request_features.py
```

Input:

```text
edge metadata
request condition
```

Output:
- feature vector đúng schema của segment model.

Rules:

```text
feature thuộc edge
    → lấy từ edge

feature thuộc request
    → lấy từ request

feature đã được upstream normalize
    → reuse

feature không nằm trong model schema
    → không tạo
```

Không tự thêm geometry-derived feature mới.

---

# 22. G11 — Predict edge energy

File:

```text
src/multihub/predict_edge_energy.py
```

Input:

```text
request-edge feature rows
segment preprocessing pipeline
segment energy model
```

Output:

```text
request_id
edge_id
predicted_segment_energy_wh
```

Sau này optional:

```text
energy_lower
energy_upper
ood_score
ood_flag
```

Bắt buộc:
- reuse exact scaler/encoder
- exact feature order
- không fit preprocessing mới trên routing scenario.

---

# 23. G12 — Enumerate candidate paths

File:

```text
src/multihub/enumerate_candidate_paths.py
```

Candidate path = một đường hợp lệ trong graph từ:

```text
source_node → destination_node
```

Candidate path:
- không phải ML feature
- không phải random route
- không sinh environmental variable mới.

Có thể dùng:

```python
networkx.all_simple_paths()
```

Filter:

```text
num_hops <= MAX_PATH_HOPS
```

và nếu cần:

```text
total_distance
    <= shortest_distance * MAX_PATH_DISTANCE_RATIO
```

Không cho path lặp node.

---

# 24. Gate: phải có nhiều path mới có route recommendation

Cho một request:

```text
n_paths = số valid source → destination paths
```

Nếu:

```text
n_paths == 0
```

→ disconnected request.

Nếu:

```text
n_paths == 1
```

→ có route nhưng không có route-choice comparison.

Nếu:

```text
n_paths >= 2
```

→ đủ điều kiện route recommendation.

Không tự tạo path giả để đủ số lượng.

---

# 25. Candidate path output

File:

```text
output/data/multihub/candidate_paths.csv
```

Columns:

```text
request_id
path_id

source_node
destination_node

node_sequence
edge_sequence

num_hops
total_distance

predicted_total_energy
predicted_ee

is_original
is_shortest
is_min_energy
is_max_ee
is_risk_aware
```

---

# 26. Path energy aggregation

Với path:

```text
P = [e1, e2, ..., ek]
```

Predicted total energy:

\[
E_{path}
=
\sum_{i=1}^{k}
E_{edge_i}
\]

Trong đó:

```text
E_edge_i
=
predicted_segment_energy_wh của edge i
```

Không cộng edge EE.

---

# 27. Path distance

\[
D_{path}
=
\sum_{i=1}^{k}
D_{edge_i}
\]

Trong đó:

```text
D_edge_i
=
segment_distance của edge i
```

Unit:
- meter.

---

# 28. Path EE

Reuse EE definition hiện có:

\[
EE_{path}
=
\frac{D_{path}}{E_{path}}
\]

Unit:

```text
m/Wh
```

Không dùng:

```text
EE_path = sum(EE_edge)
```

vì EE không additive.

---

# 29. G13 — Routing policies

File:

```text
src/multihub/routing_policies.py
```

Core policies:

```text
P0_ORIGINAL
P1_SHORTEST
P2_MIN_ENERGY
P3_MAX_EE
```

Sau này:

```text
P4_RISK_AWARE
```

---

# 30. P0 — Original

Chỉ áp dụng nếu request có historical original path mapping.

Nếu không:

```text
P0 = N/A
```

Không ép mọi request có original baseline.

---

# 31. P1 — Shortest

Score:

\[
D_{path}
\]

Chọn path có:

```text
minimum total distance
```

---

# 32. P2 — Minimum Energy

Score:

\[
E_{path}
\]

Chọn:

```text
minimum predicted total energy
```

Đây là energy-aware routing policy chính.

---

# 33. P3 — Maximum EE

Cho từng path:

1. tính `D_path`
2. tính `E_path`
3. tính `EE_path`

Chọn:

```text
maximum EE_path
```

Không dùng EE làm additive edge weight.

---

# 34. G14 — Evaluate routing

File:

```text
src/multihub/evaluate_routing.py
```

Cho mỗi request xuất:

```text
original path nếu có
shortest path
min-energy path
max-EE path
```

Mỗi path:

```text
node_sequence
distance
predicted energy
predicted EE
num_hops
```

---

# 35. Energy Saving % vs Original

Chỉ tính khi original path tồn tại.

\[
Saving\%
=
\frac{E_{original}-E_{selected}}
{E_{original}}
\times 100
\]

Kết quả phải gọi:

```text
model-predicted energy saving
```

hoặc:

```text
model-implied energy saving
```

Không gọi actual saving nếu alternative path chưa được bay như complete mission.

---

# 36. EE Gain % vs Original

Chỉ tính khi original path tồn tại.

\[
EEGain\%
=
\frac{EE_{selected}-EE_{original}}
{EE_{original}}
\times 100
\]

Main routing evaluation:
- predicted vs predicted.

Historical ground truth:
- dùng để validate model.

---

# 37. Shortest vs Min-Energy Disagreement

Với request i:

```text
disagreement_i = 1
```

nếu:

```text
shortest_path != min_energy_path
```

ngược lại:

```text
0
```

Across N eligible requests:

\[
DisagreementRate
=
\frac{\sum disagreement_i}{N}
\times 100
\]

Metric này dùng để đo:
- distance-only routing khác energy-aware routing bao nhiêu.

---

# 38. Hub Usage Rate

Với hub h:

\[
HubUsage(h)
=
\frac{
\text{số selected paths chứa hub h}
}{
\text{tổng selected paths}
}
\times 100
\]

Hub IDs lấy động từ graph.

Không hard-code tên hub.

---

# 39. Path Selection Frequency

Với unique path p:

\[
PathFrequency(p)
=
\frac{
\text{số lần policy chọn path p}
}{
\text{tổng requests}
}
\times 100
\]

Dùng để kiểm tra:
- policy có luôn chọn một path
- hay thay đổi theo request condition.

---

# 40. Main experiment không giả định lại feature

Main experiment dùng:
- operational features đã có
- environmental features đã có
- normalized feature schema hiện tại.

Không tự tạo lại:
- speed
- altitude
- payload
- weather
- relative wind

chỉ vì Multi-Hub.

---

# 41. G15 — Sensitivity Analysis

File:

```text
src/multihub/sensitivity_analysis.py
```

Sensitivity là optional extension sau core routing.

Chỉ thay đổi những feature mà:
- final model trực tiếp sử dụng
- upstream feature pipeline có thể tạo nhất quán.

Ví dụ:
- payload
- speed
- altitude
- environmental variables phù hợp.

Nếu một variable yêu cầu derived feature mà core Multi-Hub không có đủ thông tin để recompute:
- không chạy sensitivity cho variable đó trong module này.

Không thêm geometry computation mới chỉ để phục vụ sensitivity.

---

# 42. OOD / Uncertainty Extension

Chỉ thêm sau khi core graph routing chạy ổn.

OOD:
- reuse/fix RS2 implementation.

Conformal:
- reuse/fix RS2 implementation.

Edge prediction có thể thêm:

```text
energy_lower
energy_upper
ood_score
ood_flag
```

Graph topology không thay đổi.

---

# 43. Risk-Aware Path

Nếu sử dụng edge upper energy:

\[
RiskCost(path)
=
\sum U_{edge_i}
\]

Trong đó:

```text
U_edge_i
=
upper predicted energy của edge i
```

Đây chỉ là:

```text
conservative path risk score
```

Không gọi:
- calibrated route-level upper bound

trừ khi có route-level calibration riêng.

P4:

1. filter invalid/OOD edge theo policy
2. enumerate valid paths
3. tính risk cost
4. chọn minimum risk cost

---

# 44. Battery Feasibility — Optional

Chỉ implement nếu final scope vẫn cần.

Không hard-code battery capacity.

Input:
- UAV-specific battery config
- SOC nếu có
- reserve policy

Nếu no recharge:
- check total path energy.

Nếu có relay/recharge:
- là scenario extension riêng.

Không bắt buộc cho core route recommendation.

---

# 45. Output files

```text
output/data/multihub/
├── trajectory_index.csv
├── graph_nodes.csv
├── trajectory_node_mapping.csv
├── segment_level_features.csv
├── graph_edges.csv
├── route_requests.csv
├── edge_request_features.csv
├── edge_predictions.csv
└── candidate_paths.csv
```

Results:

```text
output/results/multihub/
├── policy_comparison.csv
├── policy_summary.csv
├── hub_usage.csv
├── path_frequency.csv
├── shortest_vs_energy.csv
└── sensitivity_results.csv
```

Optional:

```text
risk_aware_results.csv
```

---

# 46. File structure

```text
RS2/
├── config/
│   └── multihub_config.py
│
├── src/
│   └── multihub/
│       ├── adapters/
│       │   ├── base_adapter.py
│       │   └── ...
│       │
│       ├── load_multihub_data.py
│       ├── build_trajectory_index.py
│       ├── detect_hub_nodes.py
│       ├── cluster_hub_nodes.py
│       ├── map_trajectory_nodes.py
│       ├── segment_trajectories.py
│       ├── build_segment_features.py
│       ├── train_segment_energy_model.py
│       ├── build_route_graph.py
│       ├── build_route_requests.py
│       ├── build_edge_request_features.py
│       ├── predict_edge_energy.py
│       ├── enumerate_candidate_paths.py
│       ├── routing_policies.py
│       ├── evaluate_routing.py
│       ├── sensitivity_analysis.py
│       └── risk_aware_routing.py
│
└── run_multihub_graph.py
```

---

# 47. Run stages

```text
G00 — Load normalized data

G01 — Build trajectory index

G02 — Detect / define hub nodes

G03 — Cluster / normalize graph nodes

G04 — Map trajectory to ordered nodes

G05 — Segment trajectories

G06 — Build segment-level features

G07 — Train / validate Segment Energy-First model

G08 — Build directed graph

G09 — Build route requests

G10 — Build request-specific edge features

G11 — Predict edge energy

G12 — Enumerate candidate paths

G13 — Run routing policies

G14 — Evaluate routing

G15 — Sensitivity analysis

G16 — Integrate OOD

G17 — Integrate uncertainty

G18 — Risk-aware routing
```

---

# 48. Implementation Gates

## Gate A — Input Ready

Upstream data phải:
- cleaned
- normalized
- đúng schema.

Nếu chưa:
- stop.

## Gate B — Node / Segment Valid

Nếu trajectory không map được thành node/segment:
- sample đó không đi tiếp.

## Gate C — Segment Model Valid

Không route-score nếu:
- segment model chưa validate.

## Gate D — Graph Connected

Cho từng request:

```text
0 path
→ disconnected

1 path
→ route exists but no route-choice comparison

>= 2 paths
→ eligible for route recommendation
```

## Gate E — Feature Schema Valid

Edge-request feature vector:
- phải đúng final model schema.

---

# 49. Core plan không thay đổi khi dataset thay đổi

Nếu sau preprocessing một dataset:

- mất một số flight
- số route thay đổi
- route metadata khác
- feature optional thiếu
- topology khác
- số hub khác
- số path khác

thì chỉ ảnh hưởng:

```text
adapter
config
normalized input
runtime graph/result
```

Core pipeline vẫn giữ:

```text
load
→ nodes
→ segments
→ segment model
→ graph
→ requests
→ edge prediction
→ candidate paths
→ policy comparison
```

---

# 50. Những thứ không được hard-code

Không hard-code:

```text
route name
flight ID
hub name
source node name
destination node name
number of hubs
number of paths
dataset-specific exclusions
cleaning rules
unverified thresholds
expected model result
```

---

# 51. Core output tối thiểu

Cho mỗi routing request:

```text
source_node
destination_node
n_candidate_paths
```

Cho mỗi path:

```text
node_sequence
edge_sequence
num_hops
total_distance
predicted_total_energy
predicted_ee
```

Policy result:

```text
shortest_path
min_energy_path
max_ee_path
```

Nếu original tồn tại:

```text
original_path
saving_vs_original
ee_gain_vs_original
```

Global:

```text
eligible_request_count
disconnected_request_count
single_path_request_count

shortest_energy_disagreement_rate
hub_usage
path_selection_frequency
```

---

# 52. Tóm tắt cuối

```text
CLEAN / NORMALIZED DATA
        ↓
TRAJECTORY INDEX
        ↓
HUB / NODE DETECTION
        ↓
TRAJECTORY SEGMENTATION
        ↓
SEGMENT-LEVEL DATASET
        ↓
SEGMENT ENERGY-FIRST MODEL
        ↓
DIRECTED GRAPH
        ↓
ROUTING REQUESTS
        ↓
EDGE + REQUEST FEATURES
        ↓
PREDICT EDGE ENERGY
        ↓
ENUMERATE VALID PATHS
        ↓
┌──────────────┬──────────────┬──────────────┐
│   SHORTEST   │  MIN ENERGY  │    MAX EE    │
└──────────────┴──────────────┴──────────────┘
        ↓
ROUTE COMPARISON
        ↓
ROUTE RECOMMENDATION
        ↓
OPTIONAL:
OOD + UNCERTAINTY + RISK-AWARE
```

Điểm quan trọng:

- Không dùng `bearing`.
- Không tạo geometry feature mới nếu không cần.
- Reuse `relative_wind_angle` nếu upstream đã có đúng.
- Candidate path chỉ là đường đi hợp lệ trong graph.
- Multi-Hub core tập trung vào:
  **Hub → Segment → Graph → Path → Energy → EE → Recommendation**.
