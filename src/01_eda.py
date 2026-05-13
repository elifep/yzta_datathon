import pandas as pd
import numpy as np

# =========================
# LOAD
# =========================
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test_x.csv")

target = "bilissel_performans_skoru"

# =========================
# BASIC INFO
# =========================
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain columns:")
print(train.columns.tolist())

print("\nTarget summary:")
print(train[target].describe())

# =========================
# MISSING VALUES
# =========================
print("\nTop Missing Values - Train:")
print(train.isnull().sum().sort_values(ascending=False).head(15))

print("\nTop Missing Values - Test:")
print(test.isnull().sum().sort_values(ascending=False).head(15))

# =========================
# DATA TYPES
# =========================
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical columns:", cat_cols)
print("\nNumeric columns:", num_cols)

# =========================
# DUPLICATES
# =========================
print("\nDuplicate train rows:", train.duplicated().sum())
print("Duplicate test rows:", test.duplicated().sum())

# =========================
# CATEGORICAL SUMMARY
# =========================
print("\nCategorical distributions:")
for col in cat_cols:
    print(f"\n--- {col} ---")
    print("Unique:", train[col].nunique())
    print(train[col].value_counts(dropna=False).head(10))

# =========================
# OUTLIER CHECK (IQR)
# =========================
print("\nOutlier check:")
for col in num_cols:
    if col != target:
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((train[col] < lower) | (train[col] > upper)).sum()

        print(
            f"{col}: {outliers} outliers | "
            f"min={train[col].min()} | max={train[col].max()}"
        )

# =========================
# TARGET CORRELATION
# =========================
print("\nTarget Correlation:")
corr = train[num_cols].corr()[target].sort_values(ascending=False)
print(corr)

# =========================
# SPECIAL FEATURE CHECK
# =========================
print("\nREM + Deep Sleep summary:")
sleep_total = train["rem_yuzdesi"] + train["derin_uyku_yuzdesi"]
print(sleep_total.describe())

# =========================
# TEST UNSEEN CATEGORIES
# =========================
print("\nUnseen categories in test:")
for col in cat_cols:
    train_unique = set(train[col].dropna().unique())
    test_unique = set(test[col].dropna().unique())

    unseen = test_unique - train_unique

    print(f"{col}: {unseen}")

# =========================
# TARGET SHAPE
# =========================
print("\nTarget skewness:", train[target].skew())