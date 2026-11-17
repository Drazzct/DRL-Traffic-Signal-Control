# Evaluation Outputs

Thư mục này chứa kết quả chạy thử nghiệm và đánh giá các mô hình điều khiển tín hiệu giao thông (DRL Models & Baselines).

- **Hướng dẫn chi tiết**: Vui lòng xem tài liệu hướng dẫn đầy đủ cách đọc dữ liệu bảng và phân tích các biểu đồ tại:  
  👉 [docs/OUTPUT_EVALUATION_GUIDE.md](../docs/OUTPUT_EVALUATION_GUIDE.md)

## Cấu trúc nhanh

- `outputs/results/`:
  - `metrics.csv`: Bảng tổng hợp các chỉ số trung bình theo từng kịch bản và controller độc lập.
  - `comparison_model_vs_fixed_time.csv`: Bảng so sánh đối đầu trực tiếp (side-by-side) giữa Model và Fixed-Time kèm độ chênh lệch tuyệt đối và phần trăm cải thiện (% improvement).
  - `step_metrics.csv`: Dữ liệu đo đạc chi tiết theo từng bước điều khiển (5s/bước).
  - `step_evaluation_dashboard.png`: Dashboard 4 đồ thị theo dõi diễn biến mạng lưới giao thông.
  - `queue_comparison.png`, `waiting_time_comparison.png`, `time_loss_comparison.png`: Biểu đồ cột so sánh trực quan giữa các Controller.
  - `queue_length_timeline.png`, `accumulated_waiting_time.png`: Đồ thị chuỗi thời gian so sánh chi tiết.
- `outputs/runs/`: Nhật ký SUMO raw (`tripinfo.xml`, `emissions.xml`) cho từng phiên chạy độc lập.

