# =========================
# 1. استيراد المكتبات
# =========================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




# =========================
# 2. تحميل البيانات
# =========================

DATA_PATH = Path(__file__).resolve().parent / "data" / "student_lifestyle_100k.csv"
df = pd.read_csv(DATA_PATH)


# =========================
# 3. الفحص الأولي للبيانات
# =========================

print(df.head())
print("Shape:", df.shape)
print(df.info())
print(df.describe())


# =========================
# 4. توزيع الخصائص
# =========================

df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()


# =========================
# 5. توزيع المتغير الهدف
# =========================

target_counts = df['Depression'].value_counts()

print(target_counts)

ax = target_counts.plot(kind='bar')

for i, v in enumerate(target_counts):
    ax.text(i, v + 500, str(v), ha='center')

plt.title("Depression Target Distribution")
plt.xlabel("Depression")
plt.ylabel("Count")

plt.show()


# =========================
# 6. فحص القيم المفقودة
# =========================

print(df.isnull().sum())

# =========================
# 7. حذف العمود غير الضروري
# =========================

df.drop("Student_ID", axis=1, inplace=True)

print(df.head())
print(df.shape)

# =========================
# 8. تصنيف الأعمدة
# =========================

categorical_cols = df.select_dtypes(include=['object', 'string']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# =========================
# 9. ترميز الأعمدة الفئوية
# =========================

from sklearn.preprocessing import LabelEncoder

le_gender = LabelEncoder()
le_department = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Department'] = le_department.fit_transform(df['Department'])

print(df.head())

# =========================
# 10. ترميز المتغير الهدف
# =========================

df['Depression'] = df['Depression'].astype(int)

print(df.head())
print(df['Depression'].value_counts())

# البيانات غير متوازنة:
# الفئة 0 (غير مكتئب) تمثل تقريبًا 90%
# الفئة 1 (مكتئب) تمثل تقريبًا 10%
# لذلك سنركز لاحقًا على Recall و F1-score أكثر من Accuracy

# =========================
# 11. تقسيم الخصائص والمتغير الهدف
# =========================


X = df.drop("Depression", axis=1)
y = df["Depression"]

# أول تقسيم: 80% تدريب و20% مؤقتة
X_train, X_temp, y_train, y_temp = train_test_split(    
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ثاني تقسيم: 10% validation و10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)
# =========================
# 12. توحيد مقياس الخصائص
# =========================


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaled X_train shape:", X_train_scaled.shape)
print("Scaled X_val shape:", X_val_scaled.shape)
print("Scaled X_test shape:", X_test_scaled.shape)
# =========================
# 13. إظهار أثر التوحيد بصريًا
# =========================

plt.figure(figsize=(8, 5))

plt.hist(X_train['Physical_Activity'], bins=30)

plt.title("Before Scaling - Physical Activity")
plt.xlabel("Physical Activity")
plt.ylabel("Frequency")

plt.show()


plt.figure(figsize=(8, 5))

plt.hist(X_train_scaled[:, list(X.columns).index('Physical_Activity')], bins=30)

plt.title("After Scaling - Physical Activity")
plt.xlabel("Scaled Value")
plt.ylabel("Frequency")

plt.show()
