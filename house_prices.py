import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# -------------------------------
# 1. SET YOUR FILE PATH HERE
# -------------------------------
RAW_PATH = r"C:\Users\chand\OneDrive\Documents\DATASET\house_prices.csv"  # <- change if needed
CLEANED_OUT = "house_prices_cleaned.csv"
MODEL_OUT = "rf_regressor_baseline.joblib"

print("Loading:", RAW_PATH)
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"File not found at: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)
print("Initial shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Basic Cleaning & Imputation
# -------------------------------
print("\nMissing values (before):")
print(df.isnull().sum().sort_values(ascending=False).head(30))

# Remove any near-constant ID column if present
for id_col in ["Id", "id", "ID"]:
    if id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)
        print(f"Dropped column: {id_col}")

# Target
target = "SalePrice"
if target not in df.columns:
    raise Exception(f"Target column '{target}' not found!")

# Separate feature types
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\nNumeric cols: {len(num_cols)}  |  Categorical cols: {len(cat_cols)}")

# Impute numeric with median
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical with most frequent (mode)
cat_imputer = SimpleImputer(strategy="most_frequent")
if len(cat_cols) > 0:
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing values (after):")
print(df.isnull().sum().sort_values(ascending=False).head(30))

# -------------------------------
# 3. Encoding categorical features
# -------------------------------
# Use one-hot encoding for categorical variables (drop_first to avoid dummy trap)
if len(cat_cols) > 0:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"\nAfter one-hot, new shape: {df.shape}")

# -------------------------------
# 4. Feature scaling (numeric)
# -------------------------------
# Recompute numeric columns (after one-hot these remain numeric)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

scaler = StandardScaler()
if len(num_cols) > 0:
    df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# 5. Train/Test Split
# -------------------------------
X = df.drop(columns=[target])
y = df[target]

# If any NaNs remained in X or y (safety)
if X.isnull().any().any() or y.isnull().any():
    raise Exception("NaNs present in features/target after imputation — check pipeline.")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------------
# 6. Train model (RandomForestRegressor)
# -------------------------------
clf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# -------------------------------
# 7. Evaluation (regression metrics)
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"\nRMSE: {rmse:,.2f}")
print(f"MAE : {mae:,.2f}")
print(f"R2  : {r2:.4f}")

# Optionally show a scatter plot of true vs predicted
try:
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs Predicted (Test set)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plotting failed:", e)

# Feature importance - print top 15
try:
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    print("\nTop 15 feature importances:")
    print(feat_imp)
except Exception:
    pass

# -------------------------------
# 8. Save outputs
# -------------------------------
df.to_csv(CLEANED_OUT, index=False)
joblib.dump(clf, MODEL_OUT)

print("\nSaved cleaned data:", CLEANED_OUT)
print("Saved model:", MODEL_OUT)
print("\nDone.")
