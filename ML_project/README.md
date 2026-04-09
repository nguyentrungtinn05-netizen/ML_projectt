# 🎓 AI-based Personalized Learning Habit Analysis System

Ứng dụng web dự đoán kết quả học tập của sinh viên dựa trên thói quen học tập,
sử dụng mô hình **Stacking Ensemble** (Linear Regression + Random Forest + Gradient Boosting).

---

## 🚀 Chạy ứng dụng (1 bước duy nhất)

### Cách 1 — Double-click (Windows)
```
Double-click vào file:  run.bat
```
Trình duyệt sẽ tự động mở tại `http://localhost:8501`

### Cách 2 — Terminal
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 Yêu cầu hệ thống

| Phần mềm | Phiên bản tối thiểu |
|----------|-------------------|
| Python   | 3.9+              |
| pip      | (tự động cập nhật)|

---

## 🗂️ Cấu trúc thư mục

```
ML_project/
├── app.py                                        ← Web app chính (Streamlit)
├── model.pkl                                     ← Model đã train (tự tạo lần đầu)
├── requirements.txt                              ← Danh sách thư viện
├── run.bat                                       ← One-click launcher (Windows)
├── main.ipynb                                    ← Notebook phân tích đầy đủ
├── Student Performance Analytics Dataset.csv     ← Dataset chính (10,000 mẫu)
└── Student Performance and Learning Behavior Dataset.csv
```

---

## 🎮 Cách sử dụng ứng dụng

1. Kéo slider **Giờ học / ngày** (1-10 giờ)
2. Kéo slider **Số buổi vắng mặt** (0-20 buổi)
3. Kéo slider **Độ tập trung** (1-10)
4. Nhấn nút **"Dự đoán ngay"**
5. Xem kết quả: điểm dự đoán, xếp loại, lời khuyên và biểu đồ giờ học tối ưu

---

## 🤖 Mô hình ML

| Mô hình | Vai trò |
|---------|---------|
| Linear Regression | Base learner 1 (đơn giản, nhanh) |
| Random Forest | Base learner 2 (phi tuyến, chống overfit) |
| Gradient Boosting | Base learner 3 (boosting, chính xác cao) |
| **Stacking + Ridge** | **Meta-learner kết hợp 3 mô hình trên** |

**Dataset:** Student Performance Analytics — 10,000 sinh viên, 10 đặc trưng  
**Target:** `overall_score` (0–100)

---

## ❓ Lỗi thường gặp

| Lỗi | Giải pháp |
|-----|-----------|
| `Python not found` | Cài Python từ python.org, tick "Add to PATH" |
| `ModuleNotFoundError` | Chạy lại `run.bat` hoặc `pip install -r requirements.txt` |
| `FileNotFoundError` (CSV) | Đảm bảo file CSV nằm cùng thư mục với `app.py` |
| Port 8501 bị chiếm | Thêm `--server.port 8502` vào cuối lệnh streamlit |

---

*Đồ án cuối kỳ — AI-based Personalized Learning Habit Analysis System*
