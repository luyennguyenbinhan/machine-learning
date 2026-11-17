# Đề tài 10 — Dự đoán khách hàng đăng ký tiền gửi ngân hàng (FIT4013)

Mã nguồn tham khảo cho bài tập lớn **Nhập môn học máy — năm học 2026–2027**.

> **Phải đọc, hiểu và chạy kiểm tra trước khi nộp.** Sản phẩm này là khung triển khai để sinh viên chủ động làm thí nghiệm và viết nhận xét, không chứa kết quả giả định. Phải ghi rõ việc sử dụng AI theo hướng dẫn của giảng viên.

## 1. Bộ dữ liệu đúng theo tài liệu cô giáo

- Đề tài 10: **Dự đoán khách hàng đăng ký tiền gửi ngân hàng**.
- UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Trang UCI có hai bộ zip (`bank.zip`, `bank-additional.zip`) và nhiều phiên bản dữ liệu.
- **Dự án này chủ động chọn `bank-full.csv` từ `bank.zip`**, gồm 45.211 dòng và 16 cột đầu vào + nhãn `y`. Tài liệu của giảng viên không chỉ định cụ thể một biến thể CSV; nếu cô yêu cầu phiên bản khác, phải điều chỉnh code cho đúng.
- Biến mục tiêu `y`: `yes` (có đăng ký) / `no` (không đăng ký).
- `bank-full.csv` có dấu phân cách `;`.
- `duration` không được dùng làm đầu vào vì chỉ biết được thời lượng cuộc gọi sau khi gọi; bỏ cột này trong mô hình dự đoán trước cuộc gọi.
- `pdays = -1` biểu thị chưa từng liên lạc trong chiến dịch trước: **không được hiểu là -1 ngày**.

### Cách tải trên Windows

1. Mở https://archive.ics.uci.edu/dataset/222/bank+marketing.
2. Vào mục **Dataset Files**, tải `bank.zip` (hoặc nút Download, sau đó mở zip chứa `bank.zip`).
3. Giải nén `bank.zip`.
4. Sao chép `bank-full.csv` vào `data/` của dự án.
5. Kiểm tra đường dẫn cuối cùng: `de_tai_10_FIT4013/data/bank-full.csv`.

## 2. Cài đặt môi trường và chạy

Mở thư mục bằng VS Code, chọn **Terminal > New Terminal**, chạy:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py --data data/bank-full.csv
```

Nếu PowerShell chặn kích hoạt môi trường, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --data data/bank-full.csv
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --data data/bank-full.csv
```

## 3. Các bước tương ứng đúng đề cương

1. **Data Source:** nguồn UCI, giấy phép CC BY 4.0, ý nghĩa `y`.
2. **EDA:** kích thước, kiểu dữ liệu, NaN, `unknown`, trùng lặp, phân phối lớp, histogram tuổi, boxplot số dư.
3. **Preprocessing:** tách train/validation/test 60/20/20, phân tầng theo nhãn, bỏ `duration`, điền khuyết, chuẩn hóa số, one-hot biến phân loại.
4. **Features:** Luồng A giữ biến gốc còn hợp lệ; Luồng B bổ sung `da_lien_he_truoc`, thay xử lý `pdays = -1`, tạo nhóm tuổi. Hai luồng khác nhau từ cùng dữ liệu thô.
5. **Model:** baseline `DummyClassifier`, Gaussian Naive Bayes và Decision Tree; thử nhiều `max_depth`, `min_samples_leaf`.
6. **Evaluation:** bảng các phương án trên **validation**; mô hình được chọn theo F1 lớp `yes`; fit lại train+validation, báo cáo **test** một lần, gồm Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
7. **Error Analysis:** ghi các mẫu sai, phân tích False Negative/False Positive và ảnh hưởng thực tế.

**Chống data leakage:** `SimpleImputer`, `StandardScaler` và `OneHotEncoder` đều nằm trong `Pipeline`. Chúng chỉ `fit` trên train khi chọn mô hình, và chỉ fit lại trên train+validation trước khi đánh giá test.

## 4. Kết quả được tạo sau khi chạy

Trong thư mục `outputs/`:

- `eda_tong_quan.txt` — kích thước, loại dữ liệu, NaN, lặp, unknown, thống kê.
- `01_phan_phoi_nhan.png`, `02_phan_phoi_tuoi.png`, `03_boxplot_balance.png` — đồ thị EDA.
- `04_so_sanh_validation.csv` — so sánh mọi phương án trên validation.
- `05_so_sanh_F1_validation.png` — biểu đồ so sánh F1.
- `06_ket_qua_test.csv` — chỉ số của mô hình được chọn trên test.
- `07_phan_tich_loi.txt`, `08_confusion_matrix_test.png`, `09_cac_mau_du_doan_sai.csv` — phân tích sai sót.
- `10_mo_hinh_da_huan_luyen.joblib` và `thong_tin_mo_hinh.json` — mô hình và cấu hình.

**Không sao chép số liệu từ hướng dẫn hoặc mô hình khác.** Chèn hình/bảng do chính nhóm chạy và giải thích tại sao có kết quả đó.

## 5. Cấu trúc bài báo cáo *bắt buộc* (6 phần đúng thứ tự)

1. **Đặt vấn đề và phân tích yêu cầu**: mục tiêu, dataset, EDA, phân phối yes/no, thiếu/ngoại lệ.
2. **Thiết kế giải pháp và quy trình học máy**: sơ đồ, train/val/test, A/B, xử lý đặc trưng, thuật toán, chống leakage.
3. **Phân tích mã nguồn và chức năng**: giải thích các hàm `eda`, `build_features`, `build_model`, `evaluate`, `main`; trích code ngắn chứ không dán toàn bộ.
4. **Thử nghiệm và kết quả**: baseline, bảng validation, thí nghiệm siêu tham số, test, confusion matrix.
5. **Thảo luận và so sánh**: tại sao A khác B, NB khác Tree, FN/FP, giới hạn dữ liệu.
6. **Tổng kết và hướng phát triển**: kết quả chính và hướng cải tiến (không tự bịa độ chính xác).

Theo tài liệu, báo cáo phải viết **LaTeX và nộp PDF theo template của học phần**. Thư mục `bao_cao/` chỉ là **khung nội dung**, KHÔNG thay cho template của cô nếu template đó được phát riêng.

## 6. Nộp GitHub / bảo vệ

- Repository tối thiểu: `README.md`, `main.py`, `requirements.txt`, `bao_cao/`, hướng dẫn chạy, `AI_USAGE.md`.
- Mỗi thành viên cần commit thể hiện đóng góp và hiểu các phần mình làm.
- Dữ liệu thô có thể không đưa lên GitHub; README phải có link tải và hướng dẫn tái tạo.
- Cần hiểu: `y` là gì; vì sao bỏ `duration`; data leakage; baseline; số liệu Precision/Recall/F1; vì sao dùng stratify; lý do thử A/B và 2 thuật toán.
- Báo cáo khai báo AI theo yêu cầu; điền nội dung sử dụng thực tế vào `AI_USAGE.md`.

## 7. Trích dẫn nguồn dataset

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. License: CC BY 4.0.
