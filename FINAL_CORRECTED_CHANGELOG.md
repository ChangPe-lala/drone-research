# Báo cáo chỉnh sửa `Final.tex`

## 1. Phạm vi và nguồn xác thực

- File gốc: `C:\Users\Admin\Downloads\Final.tex`.
- File hoàn chỉnh: `Final_corrected.tex`.
- File gốc được giữ nguyên, không bị ghi đè.
- Các số flight-level được xác thực với `output/results/multiuav/validation/**/finalized`, `selected_configuration.json`, SHAP và ablation CSV.
- Các số routing được xác thực với `route_baseline_summary.csv`, `final_rq5`, topology audit, sensitivity và `paper_number_audit.json`.

## 2. Toàn bộ thay đổi từ bản hiện tại sang bản mới

### 2.1. Preamble và thông tin tác giả

1. **Bổ sung phụ thuộc TikZ rõ ràng**
   - Cũ: TikZ được sử dụng trong hình nhưng không được khai báo trực tiếp trong preamble.
   - Mới: thêm `\usepackage{tikz}` và `\usetikzlibrary{arrows.meta}`.
   - Lý do: bảo đảm hình dùng `Latex` arrow tip có thể biên dịch độc lập, không phụ thuộc vào package khác tải TikZ gián tiếp.

2. **Ẩn khung màu của hyperlink**
   - Cũ: citation, cross-reference và ORCID có khung đỏ/xanh trong PDF.
   - Mới: thêm `\hypersetup{hidelinks}`.

3. **Sửa danh sách email**
   - Cũ: `captainparrot13042005, karlpro812005 buihongochan.lodi`.
   - Mới: `captainparrot13042005, karlpro812005, buihongochan.lodi`.
   - Lý do: thiếu dấu phẩy làm hai địa chỉ Gmail bị ghép thành một chuỗi không hợp lệ.

### 2.2. Abstract

4. **Sửa mô tả feature set của DJI**
   - Cũ: `The retained DJI CatBoost model, using speed, altitude, and payload`.
   - Mới: `The retained 11-feature DJI CatBoost model, using three operational and eight environmental predictors`.
   - Lý do: manifest mới nhất chọn `FS_OPS_ENV` với 11 feature, không phải mô hình ba feature.

5. **Giảm mức khẳng định về domain shift**
   - Cũ: `indicates substantial domain shift`.
   - Mới: `is consistent with substantial domain shift`.
   - Lý do: negative holdout R² là bằng chứng phù hợp với domain shift nhưng không tự nó chứng minh quan hệ nhân quả.

6. **Làm chính xác claim về weather**
   - Cũ: `the available weather summaries add no validated incremental value`.
   - Mới: `the tested weather summaries provide no measurable incremental value under the grouped-validation design`.
   - Lý do: giới hạn kết luận trong đúng feature, dữ liệu và validation design đã thử.

7. **Sửa population của 3.62%**
   - Cũ: `14 hypothetical paths improve ... by 3.62% on average`.
   - Mới: mean 3.62% được gán cho 15 final decisions; đồng thời nói rõ 14 alternative được accept và 1 historical route được giữ lại.
   - Bổ sung: trên riêng 14 accepted alternatives, mean/median là 3.88%/4.11%, và cả 14 đều tăng predicted total energy.

### 2.3. Methodology

8. **Làm rõ số flight DJI từ nguồn đến analytical corpus**
   - Cũ: dataset có 209 flights rồi chuyển thẳng sang 194 flights.
   - Mới: public release báo cáo 209 flights; bảng được pipeline nạp có 207 flight identifiers; sau quality-control exclusions còn 194 flights và 249,011 observations.
   - Lý do: khớp `DATA_QUALITY_SUMMARY.md`, trong đó cleaning bắt đầu từ 207 và loại 13 flights.

9. **Đổi vai trò VTOL từ “test” sang “diagnose”**
   - Cũ: `used to test cross-platform behavior`.
   - Mới: `used to diagnose cross-platform behavior`.
   - Lý do: kết quả VTOL được dùng như diagnostic holdout, không chứng minh khả năng transfer thành công.

10. **Làm rõ mô hình DJI dùng 11 feature**
    - Cũ: chỉ liệt kê 3 operational và 8 environmental feature, chưa nói feature set cuối cùng.
    - Mới: bổ sung câu final DJI model dùng cả 11 predictors; grouped ablation được dùng riêng để đánh giá incremental value của environmental group.

11. **Sửa Eq. (2), phân biệt segment energy và route energy**
    - Cũ: `\hat{e}_k=\sum_{k\in r}\hat{e}_k` và mẫu số là `\hat{e}_k`.
    - Mới: `\widehat{E}(r)=\sum_{k\in r}\hat{e}_k` và `\widehat{EE}(r)=D(r)/\widehat{E}(r)`.
    - Lý do: `\hat e_k` là energy của segment k; `\widehat E(r)` mới là tổng energy của route.

12. **Gắn citation trực tiếp cho các boosting libraries được nêu trong phương pháp**
    - Cũ: chỉ citation CatBoost ở cuối câu, còn XGBoost và LightGBM có trong bibliography nhưng không được cite trực tiếp.
    - Mới: nêu CatBoost, XGBoost và LightGBM cùng các citation `catboost2018`, `xgboost2016`, `lightgbm2017` trong câu mô tả 10 model families.

### 2.4. Prediction Performance Comparison

13. **Loại phép so sánh feature-set không còn được output mới nhất hỗ trợ**
    - Cũ: compact operational versus operational-plus-weather: R² 0.8602 versus 0.7939; WMAPE 4.68% versus 5.82%.
    - Mới: mô tả đúng benchmark hiện có cho `FS_OPS_ENV`: mean R² 0.8106 và mean WMAPE 5.54% trên 10 model families.
    - Lý do: output finalized hiện không chứa benchmark tương ứng để tái tạo cặp số cũ.

14. **Cập nhật CatBoost development result**
    - Cũ: R² 0.9123, WMAPE 3.51%.
    - Mới: R² 0.9072, WMAPE 3.84%.
    - Nguồn: `dji_only/cp2_model_benchmark/model_evidence.csv`.

15. **Làm rõ ý nghĩa N trong Table II**
    - Cũ: chưa phân biệt N holdout flights và N grouped-CV segments.
    - Mới: ghi rõ ba dòng flight-level dùng N holdout flights; dòng segment dùng N segments trong grouped CV, không phải independent holdout.

16. **Cập nhật subgroup R² của combined holdout**
    - Cũ: DJI 0.9578; VTOL -0.1255.
    - Mới: DJI 0.6706 trên 48 flights; VTOL -0.3137 trên 15 flights.
    - Cách xác thực: tính lại R² trực tiếp từ `combined/finalized/holdout_predictions.csv`.
    - Giữ nguyên platform separation 96.3% vì giá trị 96.298976% vẫn đúng.

### 2.5. Feature Importance and Explainability

17. **Sửa “three-feature model” thành “11-feature model”**
    - Cũ: `For the retained three-feature DJI model`.
    - Mới: `For the retained 11-feature DJI model`.

18. **Cập nhật SHAP**
    - Speed: 3.6954 -> 3.6454 m/Wh.
    - Payload: 1.5301 -> 1.4129 m/Wh.
    - Altitude: 1.4442 -> 1.2615 m/Wh.
    - Nguồn: `shap/shap_importance.csv`.

19. **Sửa thuật ngữ và số individual ablation**
    - Cũ gọi là `Grouped ablation` khi đang mô tả loại từng feature.
    - Mới gọi đúng là `Leave-one-feature-out ablation`.
    - Speed ΔR²: 0.8022 -> 0.8480.
    - Payload ΔR²: 0.1569 -> 0.1954.
    - Altitude ΔR²: 0.1265 -> 0.1484.

20. **Cập nhật environmental group ablation**
    - Cũ: R² cải thiện 0.0332; WMAPE cải thiện 0.96 percentage points.
    - Mới: R² cải thiện 0.0021; WMAPE cải thiện 0.218 percentage points.
    - Kết luận mới giới hạn rõ: không có measurable incremental value trong grouped-validation design hiện tại.
    - Bổ sung câu giải thích: ablation được thực hiện sau model selection nên không được dùng để thay hoặc refit frozen holdout model một cách hồi tố.

### 2.6. Table III và routing interpretation

21. **Chuẩn hóa ký hiệu EE và đổi tên cột energy metric**
    - Cũ: `\widehat{EE}` và `ΔE`.
    - Mới: `\widehat{\mathrm{EE}}` và `S_E`.
    - Lý do: chuẩn hóa kiểu chữ cho ký hiệu EE và phản ánh đúng công thức energy saving.

22. **Sửa chú thích Table III**
    - Cũ: `Energy change (%) follows Eq. (4)`.
    - Mới: `Energy saving S_E (%) follows Eq. (4)`.
    - Các số trong bảng không đổi vì đã khớp `route_baseline_summary.csv`.

### 2.7. Fair Recommendation and Robustness

23. **Tách đúng hai population của EE gain**
    - Cũ: đoạn sau gọi 3.880% và 4.113% là `final EE gains` dù đây là thống kê của 14 accepted alternatives.
    - Mới:
      - 15 final decisions: mean 3.622%, median 3.800%.
      - 14 accepted alternatives: mean 3.880%, median 4.113%.
      - 1 retained historical mission đóng góp gain 0% vào population 15 decisions.

24. **Làm rõ population của energy change**
    - Cũ: `The corresponding mean and median energy changes...` chưa nêu rõ population.
    - Mới: nói rõ trên 14 accepted alternatives, mean/median energy saving là -4.621%/-4.760%; không alternative nào giảm predicted total energy.

25. **Giảm mức khẳng định của observed-baseline check**
    - Cũ: `These values support model realism`.
    - Mới: `These values provide a limited consistency check`.
    - Lý do: observed-versus-predicted check không làm hypothetical route trở thành quan sát thực tế.

### 2.8. Threats to Validity và Conclusion

26. **Sửa dấu câu**
    - Cũ: External Validity kết thúc bằng `geographic corridors` nhưng thiếu dấu chấm.
    - Mới: thêm dấu chấm cuối câu.

27. **Giảm claim ở Conclusion**
    - Cũ: `negative VTOL R² exposes domain shift hidden by the pooled score`.
    - Mới: `the negative VTOL R² is consistent with domain shift obscured by the pooled score`.
    - Lý do: cách diễn đạt mới chính xác hơn về mức độ bằng chứng.

## 3. Những phần được giữ nguyên vì đã khớp output

- Toàn bộ Table II final metrics: DJI, VTOL, combined và segment model.
- Segment model R²/WMAPE và 590 segments/194 flights.
- Graph 17 nodes, 36 directed edges.
- 147 topology ODs, 15 historical missions, 162 scenarios và 802 candidate-route records.
- Toàn bộ số P0--P4 trong Table III.
- P1=P2 trên 15/15 missions; P4 trùng P2 ở 2/15 và P3 ở 6/15.
- Observed-baseline direction agreement 12/15; WMAPE energy 6.0717% và EE 2.3106%.
- N5 14/14 historical recommendations, 33/147 topology recommendations, removal stability 75.74%, mean EE change -0.264%.
- Topology sensitivity 250/15, 200/20 và 200/10.
- Bibliography và contribution positioning.

## 4. Kiểm tra hoàn tất

- Các môi trường LaTeX `begin/end` cân bằng.
- Dấu ngoặc nhọn cân bằng.
- Có `\end{document}` hợp lệ.
- Không còn các claim/số cũ đã xác định.
- Các số cập nhật đã được assert trực tiếp với CSV/JSON mới nhất.
- Máy hiện tại không có `pdflatex`, `xelatex`, `lualatex`, `latexmk` hoặc `tectonic`; vì vậy chưa thể biên dịch lại PDF trong môi trường này. File đã được kiểm tra cấu trúc tĩnh và nên được compile một lần trên Overleaf/TeX Live trước khi nộp.
