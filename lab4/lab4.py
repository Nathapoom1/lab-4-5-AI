from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "lab2" / "Titanic-Dataset.csv"
K = 5

# อ่านข้อมูล Titanic และเลือกคอลัมน์ที่ใช้กับ KNN
df = pd.read_csv(DATA_PATH)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[features + ["Survived"]].copy()
# จัดการค่าว่างและแปลงข้อมูลข้อความเป็นตัวเลข
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"female": 0, "male": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

data = df.to_numpy(dtype=float)
test_data = data[0, :-1].copy()
actual_class = int(data[0, -1])
train_data = data[1:]

# ปรับสเกลฟีเจอร์ให้อยู่ในช่วง 0 ถึง 1 ก่อนคำนวณระยะห่าง
feature_min = train_data[:, :-1].min(axis=0)
feature_max = train_data[:, :-1].max(axis=0)
feature_range = np.where(feature_max == feature_min, 1, feature_max - feature_min)
train_features = (train_data[:, :-1] - feature_min) / feature_range
normalized_test = (test_data - feature_min) / feature_range

# คำนวณระยะห่าง Euclidean และเลือกเพื่อนบ้านที่ใกล้ที่สุด
distances = np.sqrt(np.sum((train_features - normalized_test) ** 2, axis=1))
indices = np.argsort(distances)[:K]
nearest_classes = train_data[indices, -1].astype(int)
# เลือก class ที่พบมากที่สุดในกลุ่มเพื่อนบ้าน
class_counts = np.bincount(nearest_classes, minlength=2)
predicted_class = int(np.argmax(class_counts))

print("Dataset: Titanic-Dataset.csv")
print("Features:", ", ".join(features))
print("Class: Survived (0 = ไม่รอด, 1 = รอด)")
print("Test Data:", test_data)
print("Nearest neighbor indices:", indices)
print("Nearest classes:", nearest_classes)
print("Actual class:", actual_class)
print("Predicted class:", predicted_class)
