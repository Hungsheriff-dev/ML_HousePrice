import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================== 1. Đọc dữ liệu ======================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ====================== 2. Xử lý Missing Values ======================
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

numeric_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]

for col in categorical_cols:
    train[col] = train[col].fillna("None")
    test[col] = test[col].fillna("None")

for col in numeric_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

# ====================== 3. Encode Categorical ======================
le = LabelEncoder()
for col in categorical_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined.astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ====================== 4. Chuẩn bị X, y ======================
X = train.drop(['Id', 'SalePrice'], axis=1)
y = train['SalePrice']
X_test = test.drop(['Id'], axis=1)

# ====================== 5. Phân tách dữ liệu ======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================== 6. SCALE (Bước quan trọng) ======================
scaler = StandardScaler()

# Chỉ fit trên tập train, sau đó transform cả train và validation + test
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Đã scale xong!")
print("Shape X_train_scaled:", X_train_scaled.shape)
print("Shape X_val_scaled  :", X_val_scaled.shape)
print("Shape X_test_scaled :", X_test_scaled.shape)

# ====================== 7. Train & Đánh giá 3 Model ======================

def evaluate_model(model, X_train, y_train, X_val, y_val, name): #định nghĩa hàm evaluate_model
    model.fit(X_train, y_train) #huấn luyện mô hình
    y_pred = model.predict(X_val) #y_pred: giá nhà mà mô hình dự đoán
    
    mae = mean_absolute_error(y_val, y_pred) #sai số tuyệt đối trung bình
    rmse = np.sqrt(mean_squared_error(y_val, y_pred)) #Căn bậc hai của sai số bình phương trung bình
    r2 = r2_score(y_val, y_pred) #Hệ số xác định
    
    print(f"\n===== {name} =====")
    print(f"MAE  : {mae:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print(f"R²   : {r2:.4f}")
    
    return {
        "Model": name,
        "MAE": round(mae),
        "RMSE": round(rmse),
        "R2": round(r2, 4)
    }

results = []

# 1. Linear Regression
lr = LinearRegression() #tạo mô hình hồi quy tuyến tính
results.append(evaluate_model(lr, X_train_scaled, y_train, X_val_scaled, y_val, "Linear Regression"))

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
results.append(evaluate_model(rf, X_train_scaled, y_train, X_val_scaled, y_val, "Random Forest"))

# 3. Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
results.append(evaluate_model(gb, X_train_scaled, y_train, X_val_scaled, y_val, "Gradient Boosting"))

# ====================== 8. Bảng so sánh ======================
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("BẢNG SO SÁNH CÁC MODEL")
print("="*50)
print(results_df.to_string(index=False))

# ====================== 9. Huấn luyện lại Random Forest trên toàn bộ tập train ======================

from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Scale lại trên toàn bộ tập train (quan trọng)
final_scaler = StandardScaler()
X_full_scaled = final_scaler.fit_transform(X)          # X là toàn bộ dữ liệu train đã encode
X_test_final_scaled = final_scaler.transform(X_test)   # Transform tập test

# 2. Huấn luyện Random Forest trên toàn bộ dữ liệu
final_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_full_scaled, y)

print("Đã huấn luyện xong Random Forest trên toàn bộ tập train!")
print(f"Số mẫu huấn luyện: {X_full_scaled.shape[0]}")
print(f"Số đặc trưng    : {X_full_scaled.shape[1]}")
    
# 3. Dự đoán trên tập test
test_predictions = final_model.predict(X_test_final_scaled)

print("\n5 dự đoán đầu tiên trên tập test:")
print(test_predictions[:5].round(0))

# 4. Tạo file submission
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("\nĐã lưu file submission.csv thành công!")

# 5. Lưu model và scaler để dùng lại sau này
joblib.dump(final_model, "random_forest_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
print("Đã lưu model: random_forest_model.pkl")
print("Đã lưu scaler: scaler.pkl")