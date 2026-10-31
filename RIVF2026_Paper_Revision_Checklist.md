# RIVF 2026 – Tổng hợp chỉnh sửa paper trước khi submit

**Paper:** *Platform-Aware Energy Modeling and Model-Symmetric Multi-Hub Route Screening for Delivery Drones*  
**Target:** IEEE RIVF 2026

## 1. CRITICAL – Phải kiểm tra bằng code

### 1.1. Table III – aggregation
- Kiểm tra vì sao `EE` không bằng `D / Eb` từ các giá trị hiển thị.
- Xác định chính xác cách tính `D`, `Eb`, `EE`, `ΔE`, `ΔEE`.
- Nếu `EE` được tính per-mission rồi lấy mean, phải nói rõ trong paper.
- **Câu hỏi:** Why is EE not equal to D/Eb in Table III? Show the exact code used to calculate each metric.

### 1.2. 3.62% vs 3.880% vs 4.113%
- Abstract: 3.62%.
- Results: mean 3.880%, median 4.113%.
- Xác định population của từng con số và sửa/giải thích nếu khác nhau.
- **Câu hỏi:** What exactly are the populations used to calculate 3.62%, 3.880%, and 4.113%? Show the code.

### 1.3. Pooled split 199/63
- DJI: 146 + 48 = 194.
- VTOL: 41 + 27 = 68.
- Combined: 199 + 63 = 262.
- Giải thích rằng pooled split được tạo độc lập bằng platform-date groups nếu đúng.
- Có thể thêm: “The combined-domain split is constructed independently using platform-date groups; therefore, its development and holdout counts do not equal the sums of the platform-specific partitions.”
- **Câu hỏi:** Why are the pooled development/holdout counts 199/63 instead of 187/75? Show the split code.

### 1.4. Kiểm chứng 96.3%
- Cung cấp công thức và code tính tỷ lệ pooled target sum of squares do platform-mean separation.
- **Câu hỏi:** How exactly is the 96.3% platform-mean-separation statistic calculated? Show the code.

### 1.5. Segment model R² = 0.9775
- Xác định 0.9775 là mean fold-level R² hay pooled OOF R².
- Gọi đúng là **5-fold flight-grouped cross-validated R²**, không phải independent test R².
- Nếu có thể, bổ sung independent flight-level holdout.
- Wording đề nghị: “The segment-energy model achieves a mean 5-fold flight-grouped cross-validated R² of 0.9775…”

### 1.6. P1 = P2
- P1 = shortest path; P2 = minimum predicted energy.
- Hai policy chọn cùng route ở cả 15 missions.
- Kiểm tra nguyên nhân: topology, dataset hay energy model.
- Có thể tính correlation giữa distance và predicted energy.
- **Câu hỏi:** Why do P1 and P2 select exactly the same route on all 15 missions?

### 1.7. Hypothetical routes có thực sự unseen?
- Xác định candidate routes là routes hoàn toàn unseen hay recombination của historical segments.
- Nếu là recombination, không overclaim route generalization.
- Có thể bổ sung limitation: “The routing experiment evaluates recombinations of historically observed local trajectory segments rather than entirely unseen geographic corridors.”

---

## 2. MAJOR – Nội dung khoa học

### 2.1. Làm rõ novelty
Không claim:
- new ML algorithm;
- new routing algorithm;
- minimum-energy routing;
- physical hub optimization.

Nên định vị:
> A validation-aware framework connecting empirical UAV energy prediction to bounded route screening.

Narrative:
**platform-aware validation → domain-shift audit → feature contribution analysis → additive segment-energy modeling → model-symmetric route screening**

### 2.2. Nhấn mạnh domain shift
Kết quả:
- DJI R² = 0.9299;
- VTOL R² = -0.1123;
- Combined R² = 0.9748.

Nên giải thích rằng pooled R² cao không đồng nghĩa cross-platform generalization tốt.

### 2.3. SHAP và ablation
- Speed > Payload > Altitude theo cả hai phương pháp.
- Giữ cách diễn giải là **predictive contribution**, không phải causal effect.

### 2.4. Weather
Không viết “weather is irrelevant”.
Nên dùng:
> “The available weather summaries provide no validated incremental predictive value.”

### 2.5. P3
P3 tăng predicted EE nhưng cũng tăng total predicted energy.
Phải nhấn mạnh:
> P3 is a higher-EE screening policy, not a minimum-energy or energy-saving routing policy.

### 2.6. ΔE / Energy Saving
Thống nhất terminology thành **Energy change (%)**.
Định nghĩa dấu:
- positive = energy reduction;
- negative = energy increase.

### 2.7. N5
Kết luận hiện tại hợp lý: request-specific energy preference rather than structural bottleneck.
Không tự suy diễn nguyên nhân địa lý nếu chưa có phân tích tọa độ.

### 2.8. Observed 12/15 agreement
Gọi là **supporting realism/consistency check**, không gọi là independent validation của hypothetical routes.

---

## 3. Figures / Tables / Cross-reference

Mỗi Figure/Table phải được gọi rõ trong main text.

- Table I: “Table I summarizes the datasets and evaluation protocols.”
- Table II: “Table II reports the final predictive performance.”
- Table III: “Table III compares the five route-selection policies.”
- Figure 1: “Figure 1 illustrates the constrained virtual multi-hub route-screening framework.”
- Figure 2: “Figure 2 presents the SHAP contributions…”
- Figure 3: “Figure 3 compares the historical and route-selection policies.”

Kiểm tra cross-reference của Eq. (1)–(4).

### Figure 1
- Xuất vector PDF/SVG nếu có thể.
- Nếu PNG: ≥300 DPI.
- Sửa label overlap.
- Kiểm tra ở kích thước IEEE.

### Figure 2
- Kiểm tra/tăng font trục, tick labels, legend, feature labels.

### Figure 3
- Kiểm tra font, legend, labels và readability.

---

## 4. Abbreviations

Định nghĩa/kiểm tra:
- **SHAP** = SHapley Additive exPlanations
- **CV** = cross-validation
- **OD** = origin-destination
- **GPS** = Global Positioning System
- **Wh** = watt-hour
- UAV, VTOL, EE, RQ1–RQ4, P0–P4 đã được định nghĩa.

---

## 5. Language / Typography / Notation

### `DJI_ONLY`
Thay:
> “The retained flight-level domain is therefore DJI_ONLY.”

bằng:
> “We therefore retain the DJI-only domain for subsequent flight-level analysis.”

### “high pooled score”
Thay bằng:
> “The high pooled R²…”

### “0/14 accepted routes reduce predicted Wh”
Thay bằng:
> “None of the 14 accepted routes reduces predicted total energy.”

### Equation 2/3
Kiểm tra source LaTeX:
- \hat{e}_k
- D(r)
- E_b(r)
- EE(r)
- w_EE
- w_E

### Terminology
Thống nhất:
- flight identifier / `flight_uid`;
- segment energy / segment Wh;
- predicted energy / predicted Wh;
- historical path / recorded path;
- hypothetical route / candidate route.

---

## 6. Related Work / Positioning

Không cạnh tranh trực tiếp bằng claim new routing algorithm.
Positioning nên là:
> empirical, leakage-aware, platform-aware energy-model evaluation connected to bounded route screening.

Research gap nên nhấn mạnh khoảng cách giữa:
1. energy prediction;
2. platform/domain validation;
3. route-level decision making.

---

## 7. Threats to Validity

Giữ phần hiện tại và nhấn mạnh:
- Segment model chưa có independent untouched test.
- Chỉ có 15 recorded missions cho paired recommendations.
- Candidate routes chưa được bay.
- Nếu candidate routes là recombination historical segments thì phải nói rõ.
- Graph chỉ có 17 virtual nodes / 36 directed edges từ một DJI campaign.
- Chưa mô hình hóa obstacles, controlled airspace, communication coverage, battery SOH, charging queues, physical hub costs.

---

## 8. 8 câu hỏi bắt buộc sinh viên trả lời

1. **Why is pooled R² = 0.9748 high while VTOL R² = -0.1123?**
2. **How exactly is the 96.3% platform-mean-separation statistic calculated? Show the code.**
3. **Why are the pooled splits 199/63 instead of 187/75? Show the split code.**
4. **Why is EE not equal to D/Eb in Table III? Explain the aggregation and show the code.**
5. **What is the difference between 3.62%, 3.880%, and 4.113%? Show all calculations.**
6. **Why do P1 and P2 select the same route on all 15 missions?**
7. **Is R² = 0.9775 an independent test result? If not, what exactly does it represent?**
8. **If battery preservation were the primary objective, would you select P3? Why?**

---

## 9. 5 con số sinh viên phải chứng minh bằng code

Yêu cầu đi từ **raw data → processing → intermediate result → final number** cho:
1. R² = 0.9299
2. R² = 0.9748 và 96.3%
3. R² = 0.9775 ± 0.0070
4. 3.880%
5. 75.74%

---

## 10. Final submission checklist

### Results / Code
- [ ] Verify Table III aggregation.
- [ ] Verify 3.62% / 3.880% / 4.113%.
- [ ] Verify 96.3%.
- [ ] Verify 0.9775 calculation.
- [ ] Verify 199/63 pooled split.
- [ ] Explain P1 = P2.
- [ ] Verify hypothetical route construction.
- [ ] Verify ΔE sign convention.

### Scientific writing
- [ ] Strengthen research gap.
- [ ] Clarify true novelty.
- [ ] Clarify pooled R² vs domain shift.
- [ ] Clarify P3 is not minimum-energy routing.
- [ ] Clarify segment model is CV, not independent test.
- [ ] Clarify 12/15 agreement is not route validation.
- [ ] Avoid unsupported spatial explanation of N5.

### Presentation
- [ ] Explicitly cross-reference Tables I–III.
- [ ] Explicitly cross-reference Figures 1–3.
- [ ] Check Eq. (1)–(4) references and rendering.
- [ ] Redraw Figure 1 at high resolution.
- [ ] Increase Figure 2 font if necessary.
- [ ] Check Figure 3 readability.
- [ ] Define SHAP, CV, OD, GPS, Wh.
- [ ] Fix `DJI_ONLY`.
- [ ] Standardize terminology.
- [ ] Final grammar/typo check.

---

## Final recommendation

Không cần thay đổi methodology lớn. Paper có nền tảng tốt và narrative khoa học rõ.

Tuy nhiên, chưa nên coi paper hoàn toàn sẵn sàng chỉ vì đã sửa hình thức. Trước khi submit, nhóm phải xác minh đặc biệt:
1. Table III aggregation;
2. 3.62% vs 3.880%;
3. 96.3%;
4. pooled split 199/63;
5. segment R² = 0.9775;
6. P1 = P2.

Sau khi các điểm trên được code xác nhận và giải thích rõ trong paper, cùng với việc hoàn thiện figures, cross-reference, abbreviations và notation, paper có thể được xem là **Ready for RIVF 2026 Submission**.
