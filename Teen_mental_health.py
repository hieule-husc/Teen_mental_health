import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv(r'F:\Khoa học dữ liệu\Tiểu luận\project_mental_health\data\Teen_Mental_Health_Dataset.csv',
                  encoding='utf-8-sig')

print("✅ Load thành công!")
print(f"Kích thước: {df.shape}")
print(df.head())

# %%
print("=" * 70)
print("1. KIỂM TRA THÔNG TIN CƠ BẢN")
print("=" * 70)

print(f"\nKích thước: {df.shape}")
print(f"Số dòng: {df.shape[0]}")
print(f"Số cột: {df.shape[1]}")

print("\n" + "=" * 70)
print("2. KIỂM TRA MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

print("\n" + "=" * 70)
print("3. THỐNG KÊ CƠ BẢN")
print("=" * 70)

print(df.describe())

print("\n" + "=" * 70)
print("4. KIỂM TRA KIỂU DỮ LIỆU")
print("=" * 70)

print(df.dtypes)

# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Thư viện đã import!")

# %%
print("=" * 70)
print("1. THÔNG TIN CƠ BẢN DATASET")
print("=" * 70)

print(f"\n📊 Kích thước: {df.shape}")
print(f"   - Số dòng (mẫu): {df.shape[0]}")
print(f"   - Số cột (features): {df.shape[1]}")

print(f"\n📋 Tên các cột:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print(f"\n📊 5 dòng đầu tiên:")
print(df.head())

# %%
print("\n" + "=" * 70)
print("2. KIỂM TRA GIÁ TRỊ THIẾU (MISSING VALUES)")
print("=" * 70)

missing = df.isnull().sum()
print(missing)

print(f"\n✓ Tổng missing values: {df.isnull().sum().sum()}")

if df.isnull().sum().sum() > 0:
    print("\n⚠️ Có giá trị thiếu - Cần xử lý!")
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    missing[missing > 0].plot(kind='bar', color='red', edgecolor='black')
    plt.title('Missing Values theo Cột', fontsize=14, fontweight='bold')
    plt.xlabel('Cột')
    plt.ylabel('Số lượng giá trị thiếu')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\n✅ Dữ liệu sạch - Không có missing values!")

# %%
print("\n" + "=" * 70)
print("2. KIỂM TRA GIÁ TRỊ THIẾU (MISSING VALUES)")
print("=" * 70)

missing = df.isnull().sum()
print(missing)

print(f"\n✓ Tổng missing values: {df.isnull().sum().sum()}")

if df.isnull().sum().sum() > 0:
    print("\n⚠️ Có giá trị thiếu - Cần xử lý!")
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    missing[missing > 0].plot(kind='bar', color='red', edgecolor='black')
    plt.title('Missing Values theo Cột', fontsize=14, fontweight='bold')
    plt.xlabel('Cột')
    plt.ylabel('Số lượng giá trị thiếu')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\n✅ Dữ liệu sạch - Không có missing values!")

# %%
print("\n" + "=" * 70)
print("3. KIỂM TRA DÒNG TRÙNG LẶP")
print("=" * 70)

duplicates = df.duplicated().sum()
print(f"Số dòng trùng lặp: {duplicates}")

if duplicates > 0:
    print(f"⚠️ Có {duplicates} dòng trùng lặp - Cần xóa!")
    df = df.drop_duplicates()
    print(f"✓ Sau xóa: {df.shape}")
else:
    print("✅ Không có dòng trùng lặp!")

# %%
print("\n" + "=" * 70)
print("4. THỐNG KÊ MÔ TẢ (DESCRIBE)")
print("=" * 70)

print(df.describe())

print(f"\n📊 Kiểu dữ liệu:")
print(df.dtypes)

# %%
print("\n" + "=" * 70)
print("5. PHÂN TÍCH CỘT NUMERIC")
print("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Số cột numeric: {len(numeric_cols)}")
print(f"Các cột: {numeric_cols}")

# Vẽ histogram
plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 3, i)
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Phân bố {col}', fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Tần suất')

plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("6. PHÂN TÍCH CỘT CATEGORICAL")
print("=" * 70)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Số cột categorical: {len(categorical_cols)}")
print(f"Các cột: {categorical_cols}")

# Thống kê từng cột categorical
for col in categorical_cols:
    print(f"\n📊 Cột: {col}")
    print(df[col].value_counts())
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    df[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Phân bố {col}', fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Tần suất')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
print("\n" + "=" * 70)
print("7. PHÂN TÍCH TARGET - DEPRESSION_LABEL")
print("=" * 70)

print(f"\nPhân bố depression_label:")
print(df['depression_label'].value_counts())

print(f"\nTỷ lệ:")
print(df['depression_label'].value_counts(normalize=True) * 100)

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
df['depression_label'].value_counts().plot(kind='bar', color=['green', 'red'], edgecolor='black')
plt.title('Phân bố Depression Label', fontsize=14, fontweight='bold')
plt.xlabel('Depression Label (0=No, 1=Yes)')
plt.ylabel('Tần suất')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("8. MA TRẬN TƯƠNG QUAN (CORRELATION)")
print("=" * 70)

# Chỉ chọn cột numeric
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()

print(correlation)

# Vẽ heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Teen Mental Health', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Tìm correlation cao nhất với depression_label
print(f"\n🔍 Correlation với depression_label:")
print(correlation['depression_label'].sort_values(ascending=False))

# %%
print("\n" + "=" * 70)
print("9. PHÁT HIỆN OUTLIERS (BOXPLOT)")
print("=" * 70)

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(16, 8))
numeric_df.boxplot()
plt.title('Boxplot - Phát hiện Outliers', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("10. NHẬN XÉT & KẾT LUẬN EDA")
print("=" * 70)

print(f"""
✓ TÓMLẠI EDA TEEN MENTAL HEALTH DATASET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ KỮ CỠ DỮ LIỆU:
   - {df.shape[0]} mẫu × {df.shape[1]} cột
   - Đủ để huấn luyện mô hình ML

2️⃣ CHẤT LƯỢNG DỮ LIỆU:
   - Missing values: {df.isnull().sum().sum()} (✅ Sạch)
   - Duplicate rows: {df.duplicated().sum()} (✅ Sạch)

3️⃣ CỘT NUMERIC ({len(df.select_dtypes(include=[np.number]).columns)}):
   - age, daily_social_media_hours, sleep_hours, screen_time_before_sleep
   - academic_performance, physical_activity, stress_level
   - anxiety_level, addiction_level

4️⃣ CỘT CATEGORICAL ({len(df.select_dtypes(include=['object']).columns)}):
   - gender: male, female
   - platform_usage: Instagram, TikTok, Both
   - social_interaction_level: low, high, medium

5️⃣ TARGET VARIABLE - depression_label:
   - Phân bố: {df['depression_label'].value_counts().to_dict()}
   - Loại: Binary Classification (0/1)

6️⃣ CORRELATION:
   - Các features có tương quan với depression_label
   - Sẵn sàng cho Feature Selection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ DỮ LIỆU SẴN SÀNG CHO BƯỚC TIẾP THEO: FEATURE ENGINEERING & MODELING
""")

# %%
# Nếu muốn lưu dữ liệu đã xử lý
df.to_csv(r'F:\Khoa học dữ liệu\Tiểu luận\project_mental_health\data\Teen_Mental_Health_Cleaned.csv', 
          index=False)

print("✅ Dữ liệu đã được lưu!")

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %%
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

print("=" * 70)
print("BƯỚC 4: XÂY DỰNG MÔ HÌNH HỌC MÁY")
print("=" * 70)

# Tách X (features) và y (target)
X = df.drop('depression_label', axis=1)
y = df['depression_label']

print(f"\n📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")

print(f"\nTên features:")
for i, col in enumerate(X.columns, 1):
    print(f"   {i:2d}. {col}")

# %%
print("\n" + "=" * 70)
print("1. XỬ LÝ DỮ LIỆU CATEGORICAL")
print("=" * 70)

# Tìm cột categorical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCột categorical: {categorical_cols}")

# Label Encoding cho các cột categorical
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"\n✓ {col}:")
    for i, class_name in enumerate(le.classes_):
        print(f"   {class_name} → {i}")

print(f"\n✅ Dữ liệu categorical đã xử lý!")
print(X.head())

# %%
print("\n" + "=" * 70)
print("2. CHIA DỮ LIỆU TRAIN/TEST")
print("=" * 70)

# Chia 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

print(f"\n📊 Phân bố trong Train set:")
print(y_train.value_counts())

print(f"\n📊 Phân bố trong Test set:")
print(y_test.value_counts())

# %%
print("\n" + "=" * 70)
print("3. CHUẨN HÓA DỮ LIỆU (STANDARDIZATION)")
print("=" * 70)

# Chuẩn hóa dữ liệu để các features có cùng scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Dữ liệu đã chuẩn hóa!")
print(f"Train mean: {X_train_scaled.mean(axis=0)[:5]}")
print(f"Train std: {X_train_scaled.std(axis=0)[:5]}")

# %%
print("\n" + "=" * 70)
print("4. HUẤN LUYỆN MÔ HÌNH - LOGISTIC REGRESSION")
print("=" * 70)

from sklearn.linear_model import LogisticRegression

# Tạo & huấn luyện mô hình
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

print("✅ Mô hình Logistic Regression đã huấn luyện!")

# Dự đoán trên test set
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print(f"\n✓ Dự đoán: {y_pred_log[:10]}")

# %%
print("\n" + "=" * 70)
print("5. ĐÁNH GIÁ MÔ HÌNH LOGISTIC REGRESSION")
print("=" * 70)

# Tính các metric
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)
roc_auc_log = roc_auc_score(y_test, y_pred_proba_log)

print(f"\n📊 METRIC ĐÁNH GIÁ:")
print(f"   - Accuracy (Độ chính xác): {accuracy_log:.4f}")
print(f"   - Precision (Độ chính xác dương tính): {precision_log:.4f}")
print(f"   - Recall (Tỷ lệ phát hiện): {recall_log:.4f}")
print(f"   - F1-Score: {f1_log:.4f}")
print(f"   - ROC-AUC: {roc_auc_log:.4f}")

print(f"\n🔍 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_log, target_names=['No Depression', 'Depression']))

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
print(f"\n📊 CONFUSION MATRIX:")
print(cm_log)

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("6. HUẤN LUYỆN MÔ HÌNH - RANDOM FOREST")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier

# Tạo & huấn luyện mô hình
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

print("✅ Mô hình Random Forest đã huấn luyện!")

# Dự đoán trên test set
y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

print(f"\n✓ Dự đoán: {y_pred_rf[:10]}")

# %%
print("\n" + "=" * 70)
print("7. ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST")
print("=" * 70)

# Tính các metric
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\n📊 METRIC ĐÁNH GIÁ:")
print(f"   - Accuracy: {accuracy_rf:.4f}")
print(f"   - Precision: {precision_rf:.4f}")
print(f"   - Recall: {recall_rf:.4f}")
print(f"   - F1-Score: {f1_rf:.4f}")
print(f"   - ROC-AUC: {roc_auc_rf:.4f}")

print(f"\n🔍 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_rf, target_names=['No Depression', 'Depression']))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\n📊 CONFUSION MATRIX:")
print(cm_rf)

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("8. SO SÁNH 2 MÔ HÌNH")
print("=" * 70)

# Tạo DataFrame so sánh
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [accuracy_log, precision_log, recall_log, f1_log, roc_auc_log],
    'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf]
})

print("\n")
print(comparison)

# Vẽ biểu đồ so sánh
plt.figure(figsize=(12, 6))
comparison.set_index('Metric').plot(kind='bar', width=0.8)
plt.title('So sánh Metrics - Logistic Regression vs Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.ylim([0, 1.1])
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 8))

# ROC Curve - Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={roc_auc_log:.4f})', linewidth=2)

# ROC Curve - Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_rf:.4f})', linewidth=2)

# Đường baseline
plt.plot([0, 1], [0, 1], 'k--', label='Baseline', linewidth=2)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - So sánh 2 Mô hình', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("9. FEATURE IMPORTANCE - RANDOM FOREST")
print("=" * 70)

# Lấy feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 70)
print("10. NHẬN XÉT & KẾT LUẬN")
print("=" * 70)

if roc_auc_rf > roc_auc_log:
    best_model = "Random Forest"
    best_auc = roc_auc_rf
    best_accuracy = accuracy_rf
else:
    best_model = "Logistic Regression"
    best_auc = roc_auc_log
    best_accuracy = accuracy_log

print(f"""
✓ TÓMLẠI KẾT QUẢ:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MÔ HÌNH TỐT NHẤT: {best_model}
   - Accuracy: {best_accuracy:.4f}
   - ROC-AUC: {best_auc:.4f}

📈 KẾT QUẢ LOGISTIC REGRESSION:
   - Accuracy: {accuracy_log:.4f}
   - Precision: {precision_log:.4f}
   - Recall: {recall_log:.4f}
   - F1-Score: {f1_log:.4f}
   - ROC-AUC: {roc_auc_log:.4f}

🌲 KẾT QUẢ RANDOM FOREST:
   - Accuracy: {accuracy_rf:.4f}
   - Precision: {precision_rf:.4f}
   - Recall: {recall_rf:.4f}
   - F1-Score: {f1_rf:.4f}
   - ROC-AUC: {roc_auc_rf:.4f}

🔑 TOP 5 FEATURES QUAN TRỌNG:
{feature_importance.head(5).to_string(index=False)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ MÔ HÌNH SẴN SÀNG CHO BƯỚC TIẾP THEO: DEPLOYMENT
""")

# %%
import os

# Tạo thư mục models nếu chưa có
os.makedirs(r'F:\Khoa học dữ liệu\Tiểu luận\project_mental_health\models', exist_ok=True)

print("✅ Thư mục 'models' đã được tạo!")

# %%
import joblib

# Lưu mô hình tốt nhất
if best_model == "Random Forest":
    joblib.dump(rf, r'F:\Khoa học dữ liệu\Tiểu luận\project_mental_health\models\random_forest_model.pkl')
    print("✅ Random Forest model đã lưu!")
else:
    joblib.dump(log_reg, r'F:\Khoa học dữ liệu\Tiểu luận\project_mental_health\models\logistic_regression_model.pkl')
    print("✅ Logistic Regression model đã lưu!")

# Lưu scaler
joblib.dump(scaler, r'F:\Khoa học dữ liệu\Tiểu luận\project_mental_health\models\scaler.pkl')
print("✅ Scaler đã lưu!")

# %%




