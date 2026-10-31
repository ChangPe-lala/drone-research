PLAN HOÀN CHỈNH — CHUẨN HÓA 2 DATASET UAV TỪ RAW ROS BAG TRƯỚC KHI LÀM MULTI-HUB
================================================================================

MỤC TIÊU CHÍNH
--------------
Xây dựng một pipeline dữ liệu thống nhất cho 2 bộ UAV:

DATASET 1 — DJI Matrice 100
- Dataset hiện tại của RS2.
- Đã có raw ROS bag + parameters.csv.
- Đã có pipeline xử lý từ telemetry -> weather -> flight-level features.
- Đây là baseline cần giữ nguyên tối đa.

DATASET 2 — VTOL Package Delivery Drone
- Có raw ROS bag + parameters.csv.
- Có sensor stack tương tự:
  + anemometer
  + battery
  + IMU
  + odometry
- Sẽ xử lý lại từ raw từ đầu giống Dataset 1.
- Không dùng preprocessed flights.zip làm nguồn chính cho modeling.

Mục tiêu cuối của phase dataset:
1. Hai dataset được xử lý bằng cùng logic extraction.
2. Hai dataset dùng cùng cách tính target.
3. Hai dataset giữ cùng tên cột và đơn vị nếu tương thích.
4. Không đổi tên Dataset 1 nếu không cần.
5. Chạy được:
   - DJI -> DJI
   - VTOL -> VTOL
   - DJI -> VTOL
   - VTOL -> DJI
   - DJI + VTOL combined training
6. Chọn được model/target strategy cuối trước khi bước sang Multi-Hub.

NGUYÊN TẮC CỐT LÕI
-------------------
KHÔNG:
    DJI + VTOL
        -> concat
        -> random split
        -> train
        -> báo R² cao

PHẢI:
    xử lý độc lập
        -> cùng schema
        -> cùng target
        -> within-domain
        -> cross-domain
        -> combined training

================================================================================
PHASE 0 — GIỮ NGUYÊN BASELINE RS2
================================================================================

Không phá pipeline hiện tại.

Các output hiện có của Dataset 1 giữ nguyên:

data/intermediate/flight_with_weather.csv
data/processed/cleaned_flight_data.csv
output/data/features/flight_level_features.csv
output/data/splits/train.csv
output/data/splits/test.csv

File code chính hiện tại:
src/data_processing/feature_engineering.py

Các cột flight-level đang dùng:

flight
route
date
speed
altitude
payload
flight_duration
distance
temperature_c
humidity_pct
avg_wind_speed
wind_gust_ms
wind_dir_deg
pressure_hpa
cloud_cover_pct
relative_wind_angle
battery_consumed_wh
energy_efficiency

GIỮ NGUYÊN tên các cột này.

Đơn vị Dataset 1 đang dùng:
- speed: m/s
- altitude: m
- payload: g
- flight_duration: s
- distance: m
- temperature_c: °C
- humidity_pct: %
- avg_wind_speed: m/s
- wind_gust_ms: m/s
- wind_dir_deg: degree
- pressure_hpa: hPa
- cloud_cover_pct: %
- relative_wind_angle: degree
- battery_consumed_wh: Wh
- energy_efficiency: m/Wh

CHỈ SỬA NHẸ CONFIG NẾU CẦN:
- payload unit trong feature_schema.yaml: kg -> g
- nếu speed và altitude thật sự là programmed/set values từ parameters.csv:
  deployability:
      speed    -> PRE_FLIGHT
      altitude -> PRE_FLIGHT

Không đổi tên speed/altitude thành planned_speed/planned_altitude ở phase này.

================================================================================
PHASE 1 — SẮP XẾP RAW DATASET
================================================================================

Cấu trúc khuyến nghị:

data/
├── raw/
│   ├── dji/
│   │   ├── parameters.csv
│   │   └── rosbag/
│   │       ├── 1/raw.bag
│   │       ├── 2/raw.bag
│   │       └── ...
│   │
│   └── vtol/
│       ├── parameters.csv
│       └── rosbag/
│           ├── 1/raw.bag
│           ├── 2/raw.bag
│           └── ...
│
├── intermediate/
│   ├── flight_with_weather.csv
│   └── vtol_flight_with_weather.csv
│
└── processed/
    ├── cleaned_flight_data.csv
    └── vtol_cleaned_flight_data.csv

Không sửa trực tiếp raw bag.

================================================================================
PHASE 2 — AUDIT VTOL RAW BAG TRƯỚC KHI CODE
================================================================================

Inspect ít nhất 3 loại flight VTOL:
- payload = 0
- payload = 400
- payload = 800
- pattern A/B nếu có thể

Chạy:
    rosbag info raw.bag

Confirm chính xác:
1. topic names
2. message types
3. timestamp
4. battery current sign convention
5. position coordinate convention
6. wind angle convention
7. sampling rate
8. missing topics / corrupted bag

Cần xác nhận đặc biệt:

A. BATTERY CURRENT
------------------
Kiểm tra:
- discharge current có dương hay âm?
- có spike hoặc zero dài không?

Nếu discharge current âm:
    power = voltage * abs(current)
hoặc normalize sign trước khi tích phân.

Không dùng công thức mù quáng.

B. POSITION
-----------
Dataset 1 hiện dùng position_x/y như lon/lat.

VTOL phải xác định:
- geographic lon/lat?
hay
- local ENU/XYZ meters?

Nếu lon/lat:
    dùng Haversine / 3D Haversine

Nếu local XYZ:
    dùng Euclidean 3D cumulative distance

Không ép VTOL dùng Haversine nếu tọa độ không phải địa lý.

C. WIND ANGLE
-------------
Xác nhận:
- 0° nằm hướng nào
- wind direction là FROM hay TO
- magnetic hay true north
- angle có wrap 0–360 không

================================================================================
PHASE 3 — REFACTOR ROS BAG EXTRACTOR DÙNG CHUNG
================================================================================

Không copy toàn bộ pipeline thành 2 bản.

Tách extraction logic thành module dùng chung.

Gợi ý:

src/data_processing/
├── rosbag_extractor.py
├── build_raw_dataset.py
├── feature_engineering.py
├── data_quality.py
└── ...

rosbag_extractor.py nên có:

extract_anemometer(...)
extract_battery(...)
extract_imu(...)
extract_odometry(...)
extract_rosbag(...)

Unified telemetry output:

flight
time
wind_speed
wind_angle
battery_voltage
battery_current
position_x
position_y
position_z
orientation_x
orientation_y
orientation_z
orientation_w
velocity_x
velocity_y
velocity_z
angular_x
angular_y
angular_z
linear_acceleration_x
linear_acceleration_y
linear_acceleration_z

Nếu VTOL thiếu field nào:
- chỉ giữ NaN nếu không ảnh hưởng feature engineering
- không tạo fake value

Mục tiêu:
DJI và VTOL raw bag được extract bằng cùng implementation.

================================================================================
PHASE 4 — MERGE PARAMETERS.CSV
================================================================================

Dataset 1 giữ logic hiện tại.

VTOL parameters.csv có:

flight
speed
payload
altitude
pattern
battery_discharge_rate
date
local_time
weather_station_time
Temperature
Dew Point
Humidity
Wind
Wind Speed
Wind Gust
Pressure
Precip.
Condition

Giữ các field chung:

flight      -> flight
speed       -> speed
payload     -> payload
altitude    -> altitude
date        -> date
local_time  -> time_day

Giữ platform-specific:

pattern
battery_discharge_rate
condition

KHÔNG:
pattern -> route

Vì:
pattern = flight profile / transition strategy
route = spatial route identity

Hai khái niệm khác nhau.

Nếu VTOL không có route:
    route = NaN

Không cần tạo route giả ở phase dataset.

================================================================================
PHASE 5 — WEATHER: DÙNG CÙNG SOURCE VỚI DATASET 1
================================================================================

Khuyến nghị chính:
Dùng cùng Open-Meteo/API pipeline như Dataset 1 cho VTOL.

Lợi ích:
- cùng source
- cùng units
- cùng variable names
- giảm harmonization
- dễ cross-platform hơn

Dataset VTOL weather station vẫn giữ để:
- sanity check
- comparison
- supplementary analysis

Model chính dùng weather chuẩn hóa từ cùng API.

Các cột nên giữ theo Dataset 1:

temperature_c
dew_point_c
humidity_pct
wind_speed_ms
wind_gust_ms
wind_dir_deg
precipitation_mm
pressure_hpa
cloud_cover_pct

Lưu ý:
avg_wind_speed != wind_speed_ms

avg_wind_speed:
    onboard anemometer aggregated

wind_speed_ms:
    weather API / station environmental wind

Không trộn 2 cột này.

================================================================================
PHASE 6 — BUILD VTOL TELEMETRY DATASET
================================================================================

Tạo:

data/intermediate/vtol_flight_with_weather.csv

Schema nên bám Dataset 1:

flight
time
wind_speed
wind_angle
battery_voltage
battery_current
position_x
position_y
position_z
orientation_x
orientation_y
orientation_z
orientation_w
velocity_x
velocity_y
velocity_z
angular_x
angular_y
angular_z
linear_acceleration_x
linear_acceleration_y
linear_acceleration_z
speed
payload
altitude
date
time_day

weather:
temperature_c
dew_point_c
humidity_pct
wind_speed_ms
wind_gust_ms
wind_dir_deg
precipitation_mm
pressure_hpa
cloud_cover_pct

VTOL-specific:
pattern
battery_discharge_rate
condition

Không cần đổi tên Dataset 1.

================================================================================
PHASE 7 — CLEANING DÙNG CHUNG NHƯNG KHÔNG ÉP RULE GIỐNG 100%
================================================================================

Dùng chung:
- missing checks
- duplicate timestamp checks
- invalid battery checks
- impossible time order
- sensor NaN checks
- outlier reporting

Không copy nguyên xi các rule:
- hover removal
- static removal
- speed threshold
- route exclusion

Lý do:
VTOL có:
- vertical takeoff
- transition
- fixed-wing cruise
- landing

Một đoạn low-speed có thể hợp lệ.

Tạo VTOL-specific cleaning rules nhưng cùng framework.

Output:

data/processed/vtol_cleaned_flight_data.csv

Lưu:
- flight bị loại
- lý do
- row count trước/sau
- số flight trước/sau

================================================================================
PHASE 8 — TARGET GENERATION DÙNG CHUNG 100%
================================================================================

Đây là phần quan trọng nhất.

Tạo shared functions:

calculate_flight_duration()
calculate_battery_energy()
calculate_distance()
calculate_energy_efficiency()

A. DURATION
-----------
flight_duration =
    max(time) - min(time)

B. POWER
--------
power =
    battery_voltage * normalized_battery_current

C. ENERGY
---------
Dùng timestamp thật.

Không giả định fixed dt.

Trapezoidal integration:

dt_i = t[i+1] - t[i]

avg_power_i =
    (P[i] + P[i+1]) / 2

energy_joule =
    sum(avg_power_i * dt_i)

battery_consumed_wh =
    energy_joule / 3600

D. DISTANCE
-----------
Nếu geographic:
    3D Haversine

Nếu local XYZ:
    Euclidean cumulative 3D

Output vẫn:
    distance

Unit:
    meter

E. ENERGY EFFICIENCY
--------------------
energy_efficiency =
    distance / battery_consumed_wh

Unit:
    m/Wh

Hai dataset phải dùng cùng target definition:

battery_consumed_wh
energy_efficiency

================================================================================
PHASE 9 — FLIGHT-LEVEL FEATURE ENGINEERING
================================================================================

Dataset 1:
output/data/features/flight_level_features.csv

VTOL:
output/data/features/vtol_flight_level_features.csv

VTOL flight-level tối thiểu:

flight
route
date
speed
altitude
payload
flight_duration
distance
temperature_c
dew_point_c
humidity_pct
avg_wind_speed
wind_speed_ms
wind_gust_ms
wind_dir_deg
precipitation_mm
pressure_hpa
cloud_cover_pct
relative_wind_angle
battery_consumed_wh
energy_efficiency

platform-specific:
pattern
battery_discharge_rate
condition

Nếu route không có:
route = NaN

Nếu cloud_cover_pct không có từ API:
NaN

Không fake value.

================================================================================
PHASE 10 — BỔ SUNG NHẸ DATASET 1 CHO COMMON FEATURE SET
================================================================================

Nếu flight_level_features.csv hiện chưa lưu:

dew_point_c
wind_speed_ms
precipitation_mm

thì bổ sung vào feature_engineering.py của Dataset 1.

Không đổi tên cũ.

Chỉ thêm cột.

Khuyến nghị aggregation:

dew_point_c:
    mean

wind_speed_ms:
    mean

precipitation_mm:
    mean
nếu API hourly value bị lặp trên mỗi telemetry row

Không dùng sum nếu giá trị hourly bị duplicate theo telemetry.

================================================================================
PHASE 11 — TẠO MULTI-UAV CONFIG RIÊNG
================================================================================

Không sửa ALL_FEATURES baseline hiện tại.

Tạo:

config/multiuav_config.py

Định nghĩa:

COMMON_PREFLIGHT_FEATURES = [
    "speed",
    "altitude",
    "payload",
    "temperature_c",
    "dew_point_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_dir_deg",
    "precipitation_mm",
    "pressure_hpa",
]

Tại phase dataset:
KHÔNG dùng:

flight_duration
distance
avg_wind_speed
relative_wind_angle
battery_consumed_wh
energy_efficiency

làm pre-flight input.

Lý do:
- duration: post-flight
- distance: hiện tại là actual
- avg_wind_speed: onboard measured wind
- relative_wind_angle: hiện tại có thể dựa actual flight direction
- battery_consumed_wh: target
- energy_efficiency: target derived

================================================================================
PHASE 12 — HARMONIZE 2 DATASET
================================================================================

Tạo:

src/data_processing/harmonize_uav_datasets.py

Input:
output/data/features/flight_level_features.csv
output/data/features/vtol_flight_level_features.csv

Không rename Dataset 1.

Chỉ:
- select common columns
- align dtype
- validate units
- add metadata

Thêm:

dataset_id
uav_type

Dataset 1:
dataset_id = "DJI_M100"
uav_type = "QUADCOPTER"

Dataset 2:
dataset_id = "VTOL"
uav_type = "VTOL_FIXED_WING"

Output:

output/data/features/multiuav_flight_level_features.csv

================================================================================
PHASE 13 — VALIDATE HARMONIZED DATA
================================================================================

Tạo:

src/data_processing/validate_multiuav_data.py

Check:

1. same units
2. target > 0
3. missing %
4. duplicate flight IDs within dataset
5. range comparison
6. date parse
7. invalid weather
8. impossible battery values
9. distance sanity
10. energy sanity

Tạo report:

feature
DJI_min
DJI_max
VTOL_min
VTOL_max
overlap
missing_DJI
missing_VTOL

Các feature quan trọng:

speed
altitude
payload
temperature_c
humidity_pct
wind_speed_ms
wind_gust_ms
pressure_hpa
battery_consumed_wh
energy_efficiency
distance
flight_duration

================================================================================
PHASE 14 — EXPERIMENT 1: WITHIN-DJI
================================================================================

Feature:
COMMON_PREFLIGHT_FEATURES

Target chính:
battery_consumed_wh

Split:
group by date

Models:
reuse shortlist hiện có:
- Linear Regression / Ridge
- ExtraTrees
- XGBoost
- CatBoost

Metrics:
MAE
RMSE
R²
WMAPE
Bias
MedAE

Mục tiêu:
baseline Energy-First trên DJI bằng common feature set.

================================================================================
PHASE 15 — EXPERIMENT 2: WITHIN-VTOL
================================================================================

Same:
COMMON_PREFLIGHT_FEATURES
battery_consumed_wh

Split:
GroupKFold by date / campaign

Không random row split.

Mục tiêu:
xem cùng methodology có hoạt động trên VTOL không.

================================================================================
PHASE 16 — EXPERIMENT 3: DJI -> VTOL
================================================================================

Train:
DJI only

Hyperparameter tuning:
DJI only

Validation:
DJI only

External test:
VTOL

Không dùng VTOL để chọn model.

Metrics:
MAE
RMSE
R²
WMAPE
Bias

Đây là external cross-platform test quan trọng.

Nếu performance thấp:
KHÔNG xem là thất bại.

Đó là evidence:
cross-platform domain shift.

================================================================================
PHASE 17 — EXPERIMENT 4: VTOL -> DJI
================================================================================

Train/tune:
VTOL only

Test:
DJI

Mục tiêu:
reverse transfer.

So sánh với DJI -> VTOL.

================================================================================
PHASE 18 — DIRECT EE VS ENERGY-FIRST
================================================================================

Chạy trên cả 2 platform.

Strategy A:
X -> energy_efficiency

Strategy B:
X -> battery_consumed_wh

Đối với Energy-First:
chỉ derive EE khi có distance context hợp lệ.

Không dùng actual test-flight distance như pre-flight input.

Mục tiêu:
xem Energy-First có:
- thấp RMSE hơn
- ổn định hơn
- transfer tốt hơn
- ít bias hơn

Không kết luận dựa trên R² duy nhất.

================================================================================
PHASE 19 — DOMAIN SHIFT ANALYSIS
================================================================================

So sánh DJI vs VTOL:

speed
altitude
payload
temperature_c
humidity_pct
wind_speed_ms
wind_gust_ms
pressure_hpa
distance
energy
EE

Dùng:
- summary statistics
- histogram
- KS test
- Wasserstein distance
- standardized mean difference

Mục tiêu:
xác định vì sao cross-platform error tăng.

Không chỉnh data để ép distribution giống nhau.

================================================================================
PHASE 20 — COMBINED TRAINING
================================================================================

Chỉ chạy sau 4 experiment trên.

Train:
DJI + VTOL

Feature:
COMMON_PREFLIGHT_FEATURES

Target:
battery_consumed_wh

Group:
dataset_id + "_" + date

So sánh:

DJI-only
VTOL-only
Combined

Mục tiêu:
xem heterogeneous training có tăng robustness không.

================================================================================
PHASE 21 — UNIVERSAL VS PLATFORM-AWARE
================================================================================

MODEL U — Universal
-------------------
Input:
COMMON_PREFLIGHT_FEATURES

Không dùng:
uav_type
pattern
battery_discharge_rate

MODEL P — Platform-aware
------------------------
Input:
COMMON_PREFLIGHT_FEATURES
+
uav_type

Optional VTOL ablation:
pattern
battery_discharge_rate

Mục tiêu:
đo ảnh hưởng của platform identity.

Không dùng pattern trong universal model.

================================================================================
PHASE 22 — MODEL SELECTION CUỐI PHASE DATASET
================================================================================

Không chọn final model chỉ vì DJI R² cao nhất.

Final model phải cân bằng:

1. DJI within-domain
2. VTOL within-domain
3. DJI -> VTOL
4. VTOL -> DJI
5. combined performance
6. WMAPE
7. Bias
8. deployability

Ưu tiên:
Energy-First model nếu evidence ủng hộ.

Target:
battery_consumed_wh

================================================================================
PHASE 23 — OOD VÀ CONFORMAL SAU KHI DATASET ỔN
================================================================================

Chưa làm trước khi hoàn tất phase dataset.

Sau khi model cuối được khóa:

OOD:
- train ID detector trên train domain
- test unseen same-platform
- test other platform
- đo relation giữa OOD score và prediction error

Conformal:
- calibration group-aware theo date
- evaluate coverage within-domain
- evaluate coverage cross-platform

Đây là phase robustness sau dataset.

================================================================================
PHASE 24 — CHỈ SAU ĐÓ MỚI LÀM MULTI-HUB
================================================================================

Điều kiện để chuyển:

[ ] 2 raw pipelines ổn định
[ ] target same definition
[ ] units thống nhất
[ ] Dataset 1 baseline không bị phá
[ ] VTOL flight-level dataset sạch
[ ] common pre-flight feature set khóa
[ ] DJI -> DJI có result
[ ] VTOL -> VTOL có result
[ ] DJI -> VTOL có result
[ ] VTOL -> DJI có result
[ ] Direct EE vs Energy-First có result
[ ] Domain shift có result
[ ] Combined training có result
[ ] Final energy model được chọn
[ ] Không có target leakage

Sau đó mới:

FINAL ENERGY MODEL
        ->
PLANNED ROUTE FEATURES
        ->
SYNTHETIC MULTI-HUB TESTBED
        ->
CANDIDATE ROUTES
        ->
ENERGY PREDICTION
        ->
UNCERTAINTY/OOD
        ->
RISK-AWARE ROUTE RECOMMENDATION

================================================================================
FILE STRUCTURE CUỐI GỢI Ý — MINIMAL CHANGE
================================================================================

RS2/
├── config/
│   ├── dataset_config.py
│   ├── feature_schema.yaml
│   ├── research_config.py
│   └── multiuav_config.py                  # NEW
│
├── data/
│   ├── raw/
│   │   ├── dji/
│   │   │   ├── parameters.csv
│   │   │   └── rosbag/
│   │   └── vtol/
│   │       ├── parameters.csv
│   │       └── rosbag/
│   │
│   ├── intermediate/
│   │   ├── flight_with_weather.csv
│   │   └── vtol_flight_with_weather.csv
│   │
│   └── processed/
│       ├── cleaned_flight_data.csv
│       └── vtol_cleaned_flight_data.csv
│
├── src/
│   ├── data_processing/
│   │   ├── rosbag_extractor.py             # SHARED
│   │   ├── build_raw_dataset.py
│   │   ├── build_vtol_raw_dataset.py       # thin wrapper if needed
│   │   ├── feature_engineering.py
│   │   ├── vtol_feature_engineering.py     # thin wrapper if needed
│   │   ├── harmonize_uav_datasets.py
│   │   └── validate_multiuav_data.py
│   │
│   └── modeling/
│       ├── cross_dataset_evaluation.py
│       ├── target_strategy.py
│       └── combined_training.py
│
├── output/
│   ├── data/
│   │   └── features/
│   │       ├── flight_level_features.csv
│   │       ├── vtol_flight_level_features.csv
│   │       └── multiuav_flight_level_features.csv
│   │
│   ├── results/
│   │   └── multiuav/
│   └── figures/
│       └── multiuav/
│
├── run_research.py
└── run_dataset_extension.py                # NEW

================================================================================
RUN_DATASET_EXTENSION STAGES
================================================================================

D00 — VTOL Raw Audit
D01 — Raw ROS Extraction
D02 — Parameter + Weather Merge
D03 — VTOL Cleaning
D04 — Shared Target Generation
D05 — Flight-Level Feature Build
D06 — Dataset Harmonization
D07 — Within-Domain Evaluation
D08 — Cross-Platform Evaluation
D09 — Direct EE vs Energy-First
D10 — Domain Shift Analysis
D11 — Combined Training
D12 — Universal vs Platform-Aware
D13 — Final Multi-UAV Model Selection

Không sửa numbering E00-E13 hiện tại.

================================================================================
KẾT QUẢ BẮT BUỘC CẦN XUẤT
================================================================================

1. Dataset summary table
------------------------
Dataset
UAV
Flights
Dates
Speed range
Payload range
Altitude range
Wind range
Distance range
Energy range

2. Within-domain table
----------------------
Model
DJI MAE/RMSE/R²/WMAPE
VTOL MAE/RMSE/R²/WMAPE

3. Cross-domain table
---------------------
Model
DJI -> VTOL
VTOL -> DJI

4. Target strategy table
------------------------
Direct EE
Energy-First

5. Combined training table
--------------------------
DJI-only
VTOL-only
Combined

6. Universal vs Platform-aware table
------------------------------------
Universal
+ UAV type
+ platform-specific optional

7. Domain shift report
----------------------
feature distributions
KS
Wasserstein
SMD

================================================================================
SUCCESS CRITERIA
================================================================================

Phase dataset hoàn thành khi có thể trả lời:

RQ-D1:
Cùng một pre-flight feature set có dự đoán energy trên cả DJI và VTOL không?

RQ-D2:
Model train DJI transfer sang VTOL tốt tới đâu?

RQ-D3:
Model train VTOL transfer sang DJI tốt tới đâu?

RQ-D4:
Energy-First có generalize tốt hơn Direct EE không?

RQ-D5:
Combined heterogeneous training có cải thiện robustness không?

RQ-D6:
UAV identity có cần thiết để model dự đoán tốt không?

RQ-D7:
Feature/domain nào gây cross-platform shift mạnh nhất?

Chỉ khi trả lời được các câu trên mới chuyển sang Multi-Hub.

================================================================================
TÓM TẮT THỰC THI
================================================================================

DJI raw.bag
   +
DJI parameters.csv
   |
   v
SHARED ROS EXTRACTION
   |
   v
DJI telemetry + weather
   |
   v
DJI flight-level


VTOL raw.bag
   +
VTOL parameters.csv
   |
   v
SAME SHARED ROS EXTRACTION
   |
   v
VTOL telemetry + weather
   |
   v
VTOL flight-level


DJI flight-level
       +
VTOL flight-level
       |
       v
COMMON SCHEMA
       |
       v
DJI -> DJI
VTOL -> VTOL
DJI -> VTOL
VTOL -> DJI
       |
       v
DIRECT EE vs ENERGY-FIRST
       |
       v
DOMAIN SHIFT
       |
       v
COMBINED TRAINING
       |
       v
FINAL MULTI-UAV ENERGY MODEL
       |
       v
OOD / CONFORMAL
       |
       v
MULTI-HUB SIMULATION
