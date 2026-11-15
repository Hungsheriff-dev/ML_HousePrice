# House Price Prediction

Dự án Machine Learning dự đoán giá nhà sử dụng dataset **House Prices - Advanced Regression Techniques** từ Kaggle.

## 1. Bài toán

Xây dựng mô hình hồi quy để dự đoán giá bán nhà (`SalePrice`) dựa trên các đặc trưng của căn nhà (diện tích, chất lượng, vị trí, năm xây dựng...).

## 2. Dữ liệu

- Nguồn: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Tập train: 1460 căn nhà, 81 cột
- Tập test: 1459 căn nhà

## 3. Các bước thực hiện

1. Đọc và khám phá dữ liệu
2. Xử lý giá trị thiếu (Missing Values)
3. Mã hóa biến phân loại (Label Encoding)
4. Chuẩn hóa dữ liệu số (StandardScaler)
5. Huấn luyện và so sánh 3 mô hình:
   - Linear Regression
   - Random Forest
   - Gradient Boosting
6. Chọn mô hình tốt nhất (Random Forest)
7. Huấn luyện lại trên toàn bộ tập train
8. Dự đoán trên tập test và tạo file submission

## 4. Kết quả

| Model              | MAE     | RMSE    | R²     |
|--------------------|---------|---------|--------|
| Linear Regression  | 21,809  | 34,885  | 0.8413 |
| **Random Forest**  | **17,584** | **28,319** | **0.8954** |
| Gradient Boosting  | 17,755  | 30,691  | 0.8772 |

→ **Random Forest** cho kết quả tốt nhất.

## 5. Cách chạy

### Cài đặt thư viện

```bash
pip install pandas numpy scikit-learn joblib