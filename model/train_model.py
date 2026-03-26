import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv("../data/organic_crop_data_with_target.csv")
df.dropna(inplace=True)

# 2. Encode categorical column
le_crop = LabelEncoder()
df["Crop_Name"] = le_crop.fit_transform(df["Crop_Name"])

# 3. Encode target
df["Suitable_For_Weather"] = df["Suitable_For_Weather"].map({
    "Yes": 1,
    "No": 0
})

# 4. Features & target
X = df[["Crop_Name", "Temperature_Min_C", "Temperature_Max_C"]]
y = df["Suitable_For_Weather"]

# 5. Train–Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(acc * 100, 2), "%")
print("Confusion Matrix:\n", cm)

# 8. Save model & encoder
joblib.dump(model, "organic_rf_model.pkl")
joblib.dump(le_crop, "crop_label_encoder.pkl")

print("TRAIN–TEST DONE & MODEL SAVED")


print("Train data:", X_train.shape[0])
print("Test data:", X_test.shape[0])
