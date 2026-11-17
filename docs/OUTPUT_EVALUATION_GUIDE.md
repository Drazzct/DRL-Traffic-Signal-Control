# Hướng Dẫn Đọc Dữ Liệu và Biểu Đồ Đánh Giá (Evaluation Output Guide)

Tài liệu này cung cấp hướng dẫn chi tiết về cấu trúc thư mục `outputs/`, ý nghĩa các trường dữ liệu trong các file CSV/XML và cách đọc, phân tích các biểu đồ trực quan hóa được sinh ra sau quá trình thử nghiệm đánh giá (evaluation) bộ điều khiển đèn giao thông DRL và các giải thuật Baseline.

---

## 1. Cấu Trúc Tổng Quan Thư Mục `outputs/`

```text
outputs/
├── results/                                  # Kết quả đánh giá & biểu đồ trực quan hóa
│   ├── metrics.csv                           # Bảng tổng hợp chỉ số hiệu năng cấp Episode
│   ├── step_metrics.csv                      # Chuỗi thời gian đo đạc động học theo từng bước (Step-level)
│   ├── waiting_time_comparison.png           # Biểu đồ so sánh thời gian chờ trung bình
│   ├── queue_comparison.png                  # Biểu đồ so sánh độ dài hàng đợi trung bình
│   ├── time_loss_comparison.png              # Biểu đồ so sánh tổn thất thời gian trung bình
│   ├── step_evaluation_dashboard.png         # Dashboard 4 đồ thị theo dõi diễn biến mạng lưới
│   ├── accumulated_waiting_time.png          # Đồ thị tích lũy thời gian chờ theo thời gian
│   └── queue_length_timeline.png             # Đồ thị biến thiên độ dài hàng đợi theo thời gian
└── runs/                                     # Dữ liệu SUMO raw cho từng phiên chạy độc lập
    └── <run_id>/
        ├── tripinfo/
        │   ├── tripinfo.xml                  # Nhật ký chuyến đi chi tiết của từng phương tiện
        │   └── emissions.xml                 # Dữ liệu tiêu hao nhiên liệu và phát thải khí
        └── results/                          # Báo cáo phụ trợ theo run
```

---

## 2. Ý Nghĩa Các File Dữ Liệu Bảng (Tabular Data)

### 2.1. File `outputs/results/metrics.csv`
Lưu trữ các chỉ số tổng hợp cấp độ tập mô phỏng (Episode-level) để so sánh hiệu năng tổng quát giữa các Controller trên cùng một kịch bản giao thông và Seed ngẫu nhiên.

| Tên Cột | Đơn Vị | Ý Nghĩa Kỹ Thuật | Đánh Giá Tốt Khi |
| :--- | :---: | :--- | :---: |
| `controller` | Chuỗi | Tên bộ điều khiển (vd: `dqn_agent`, `fixed_time`, `random`). | - |
| `scenario_id` | Chuỗi | Tên kịch bản giao thông (vd: `01_morning_peak_inbound.rou`). | - |
| `seed` | Số nguyên | Seed ngẫu nhiên được dùng để cố định hành vi dòng xe. | - |
| `average_waiting_time` | Giây (s) | **Thời gian chờ trung bình** của mỗi phương tiện (tính khi vận tốc $< 0.1\text{ m/s}$). | Càng thấp càng tốt $\downarrow$ |
| `average_queue_length` | Xe (veh) | **Độ dài hàng đợi trung bình** tức thời toàn nút giao qua các bước. | Càng thấp càng tốt $\downarrow$ |
| `time_loss` | Giây (s) | **Tổn thất thời gian trung bình** do phải giảm tốc hoặc dừng chờ so với chạy tự do (free-flow speed). | Càng thấp càng tốt $\downarrow$ |
| `throughput` | Xe/giờ (veh/h) | **Thông lượng hoàn thành**: Số lượng xe thoát khỏi mạng lưới quy đổi theo giờ. | Càng cao càng tốt $\uparrow$ |
| `phase_switch_rate` | Tỷ lệ $[0, 1]$ | Tỷ lệ đổi pha tín hiệu giữa hai bước điều khiển liên tiếp ($= \frac{\text{Số lần đổi pha}}{\text{Tổng số bước}}$). | Cân bằng (tránh đổi quá dồn dập gây gián đoạn dòng xe) |
| `min_green_violations`| Số lần | Số lần thuật toán vi phạm ràng buộc thời gian xanh tối thiểu ($T_{min\_green}$). | Bằng $0$ |
| `travel_time` | Giây (s) | **Thời gian hành trình trung bình** từ lúc xuất phát đến lúc rời khỏi mạng lưới. | Càng thấp càng tốt $\downarrow$ |
| `fuel` | ml hoặc mg | Tổng tiêu hao nhiên liệu của tất cả phương tiện trong quá trình mô phỏng. | Càng thấp càng tốt $\downarrow$ |
| `co2` | mg | Tổng lượng khí thải Carbon Dioxide ($CO_2$) sinh ra. | Càng thấp càng tốt $\downarrow$ |
| `nox` | mg | Tổng lượng khí thải Oxit Nitơ ($NO_x$) độc hại sinh ra. | Càng thấp càng tốt $\downarrow$ |
| `inference_latency` | Giây (s) | Thời gian suy luận trung bình của mô hình (Model Predict Latency) ở mỗi bước. | Càng thấp càng tốt $\downarrow$ (thường $< 0.01\text{ s}$) |

---

### 2.2. File `outputs/results/comparison_model_vs_fixed_time.csv`
Bảng so sánh đối chứng song song (Side-by-Side Comparison) trực tiếp giữa mô hình (DQN Agent) và baseline Fixed-Time cho từng kịch bản, tự động tính độ lệch (`_diff`) và tỷ lệ phần trăm cải thiện (`_improvement_pct`). Cuối bảng có dòng `OVERALL_AVERAGE` tính trung bình toàn bộ kịch bản.

- **Chỉ số thời gian & hàng đợi** (`waiting_time`, `queue_length`, `time_loss`, `travel_time`): Tỷ lệ cải thiện dương ($> 0\%$) thể hiện mô hình giúp **giảm thời gian chờ / giảm tắc nghẽn**.
- **Thông lượng** (`throughput`): Tỷ lệ cải thiện dương ($> 0\%$) thể hiện mô hình giúp **tăng lưu lượng xe thoát nút giao**.
- **Mức tiêu hao & phát thải** (`fuel`, `co2`, `nox`): Tỷ lệ cải thiện dương ($> 0\%$) thể hiện mô hình giúp **tiết kiệm nhiên liệu và giảm ô nhiễm**.

---

### 2.3. File `outputs/results/step_metrics.csv`
Ghi lại trạng thái vi mô của mạng lưới sau mỗi chu kỳ điều khiển $\Delta t$ (mặc định 5 giây/step). Đây là cơ sở dữ liệu để vẽ biểu đồ diễn biến động học theo thời gian.

| Tên Cột | Đơn Vị | Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :--- |
| `step` | Giây (s) | Mốc thời gian mô phỏng SUMO hiện tại (0s, 5s, 10s, 15s,...). |
| `controller` | Chuỗi | Bộ điều khiển đang vận hành ở bước này. |
| `scenario_id` | Chuỗi | Kịch bản luồng xe đang chạy. |
| `seed` | Số nguyên | Seed ngẫu nhiên. |
| `episode` | Số nguyên | Chỉ số tập chạy lặp lại. |
| `queue_length` | Xe (veh) | Tổng số xe đang dừng chờ tại thời điểm bước này. |
| `waiting_time` | Giây (s) | Thời gian chờ tích lũy của các xe đang dừng trong bước hiện tại. |
| `accumulated_waiting_time` | Giây (s) | Tổng tích lũy dồn thời gian chờ từ đầu mô phỏng đến hiện tại. |
| `mean_speed` | m/s | Vận tốc trung bình của tất cả xe đang lưu thông trên toàn nút giao. |
| `reward` | Điểm số | Giá trị phần thưởng (Reward) mà agent nhận được tại bước này. |
| `cumulative_reward` | Điểm số | Tổng tích lũy phần thưởng của agent từ bước 0 đến hiện tại. |
| `action` | Số nguyên | Quyết định hành động (Action/Phase ID) mà mô hình đã chọn. |

---

### 2.3. Các File SUMO Raw trong `outputs/runs/<run_id>/`
- **`tripinfo.xml`**: Lưu chi tiết từng chuyến xe đã hoàn thành. Chứa các thẻ `<tripinfo id="..." depart="..." arrival="..." duration="..." routeLength="..." waitingTime="..." timeLoss="..."/>`.
- **`emissions.xml`**: Lưu trữ phát thải theo từng phương tiện và từng bước thời gian dựa trên mô hình phát thải HBEFA3 của SUMO.

---

## 3. Hướng Dẫn Đọc Các Biểu Đồ Trực Quan Hóa (Plots)

### 3.1. Dashboard Tổng Hợp: `step_evaluation_dashboard.png`
File dashboard này gồm **4 đồ thị con (2x2 grid)**, biểu diễn toàn cảnh động học điều khiển:

1. **Góc trên - bên trái: `Queue Length over Time`**
   - **Trục X**: Thời gian mô phỏng (giây).
   - **Trục Y**: Số lượng xe dừng chờ (chiều dài hàng đợi tức thời).
   - **Cách đọc**:
     - Đường đồ thị càng thấp và càng ít gai nhọn (spikes) chứng tỏ nút giao thông thoáng, xe không bị dồn ứ lâu.
     - Nếu đường đồ thị dốc đứng tăng liên tục: Báo hiệu ùn tắc hoặc tín hiệu đèn không kịp giải tỏa lưu lượng xe đến.
2. **Góc trên - bên phải: `Accumulated Waiting Time over Time`**
   - **Trục X**: Thời gian mô phỏng (giây).
   - **Trục Y**: Tổng thời gian chờ tích lũy dồn (giây).
   - **Cách đọc**:
     - Luôn có xu hướng dốc lên. Đường của Controller nào có độ dốc (slope) thấp hơn nghĩa là thuật toán đó làm giảm tốc độ tích lũy chậm trễ tốt hơn.
3. **Góc dưới - bên trái: `Average Traffic Speed over Time`**
   - **Trục X**: Thời gian mô phỏng (giây).
   - **Trục Y**: Vận tốc trung bình (m/s) của dòng xe.
   - **Cách đọc**:
     - Vận tốc càng cao và tiệm cận tốc độ thiết kế (13.89 m/s $\approx$ 50 km/h) thì dòng xe lưu thông càng thuận lợi.
     - Vận tốc sụt giảm sâu xuống gần 0 m/s thể hiện tình trạng kẹt xe cục bộ.
4. **Góc dưới - bên phải: `Cumulative Reward over Time`**
   - **Trục X**: Thời gian mô phỏng (giây).
   - **Trục Y**: Điểm phần thưởng tích lũy của Agent.
   - **Cách đọc**:
     - Phản ánh hàm mục tiêu tối ưu hóa của DRL (thường là số âm do phạt hàng đợi, phạt delay và phạt chuyển pha).
     - Đường dốc thoải hơn (ít âm hơn) cho thấy Agent nhận ít hình phạt hơn, hành vi điều khiển khớp với mục tiêu tối ưu.

---

### 3.2. Biểu Đồ So Sánh Cột (Bar Charts)

- **`queue_comparison.png` (So sánh hàng đợi trung bình)**:
  - Thể hiện sự chênh lệch số lượng xe kẹt trung bình tại nút giữa DRL Model và Baseline Fixed-Time.
  - Cột càng ngắn $\rightarrow$ Hiệu quả giảm chiều dài hàng đợi càng cao.
- **`waiting_time_comparison.png` (So sánh thời gian chờ trung bình)**:
  - Cho biết trung bình một phương tiện phải dừng chết bao nhiêu giây trước khi qua giao lộ.
  - Cột thấp hơn thể hiện phương tiện được thông hành nhanh hơn.
- **`time_loss_comparison.png` (So sánh tổn thất thời gian)**:
  - So sánh độ trễ tổng thể bao gồm cả thời gian hãm phanh chậm lại, chờ đèn và tăng tốc lại.
  - Phản ánh mức độ trơn tru và liên tục của luồng giao thông.

---

### 3.3. Biểu Đồ Dạng Đường Theo Thời Gian (Timelines)

- **`queue_length_timeline.png`**:
  - So sánh trực tiếp đường biến thiên hàng đợi của từng Controller trên cùng một hệ trục. Cho thấy Controller nào phản ứng nhanh hơn khi làn sóng phương tiện (shockwave/peak) ập vào.
- **`accumulated_waiting_time.png`**:
  - So sánh sự phân kỳ (divergence) của tổng độ trễ. Khoảng cách giữa 2 đường càng nới rộng theo thời gian chứng tỏ sự vượt trội của giải thuật tốt hơn.

---

## 4. Phân Tích Đánh Đổi (Trade-offs) Khi Đánh Giá

Khi đối chiếu giữa mô hình học tăng cường (**DRL Agent**) và bộ đèn cố định chu kỳ (**Fixed-Time**), cần lưu ý các yếu tố cân bằng:

1. **Hàng đợi (Queue) vs Tần suất chuyển pha (Phase Switch Rate)**:
   - DRL Agent có khả năng linh hoạt chuyển pha để cắt ngắn hàng đợi trên các nhánh đang dồn xe (giúp `average_queue_length` giảm sâu).
   - Tuy nhiên, nếu chuyển pha quá thường xuyên (`phase_switch_rate` cao), thời gian chuyển tiếp vàng + đỏ dọn đường (clearance phase) tăng lên, có thể làm `time_loss` tăng nhẹ.
2. **Khí thải và Tiêu hao nhiên liệu (Emissions & Fuel)**:
   - Các xe dừng rồi khởi động lại nhiều lần (stop-and-go) sẽ tiêu tốn nhiên liệu và xả nhiều $CO_2, NO_x$ hơn xe duy trì vận tốc đều.
   - Một Controller lý tưởng là Controller vừa giảm được hàng đợi, vừa giữ cho dòng xe chuyển động mượt mà ít giật cục.
3. **Thời gian xanh tối thiểu (`min_green_violations`)**:
   - Trong giao thông thực tế, đèn xanh một nhánh phải duy trì ít nhất 10–15 giây để đảm bảo an toàn cho xe giải tỏa. Cột này luôn bắt buộc phải bằng **0**.

