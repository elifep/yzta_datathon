import os
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# =========================
# CONFIG
# =========================
DATA_DIR = "data"
SUBMISSION_DIR = "submissions"

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test_x.csv"

TARGET = "bilissel_performans_skoru"

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# =========================
# MISSING VALUE CLEANING
# =========================
for col in train.columns:
    if col == TARGET:
        continue

    if train[col].dtype == "object":
        train[col] = train[col].fillna("Bilinmiyor")
        if col in test.columns:
            test[col] = test[col].fillna("Bilinmiyor")
    else:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    df = df.copy()

    df["uyku_kalite_skoru"] = df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]

    df["uyku_bozulma_skoru"] = (
        df["uykuya_dalma_suresi_dk"] +
        df["gecelik_uyanma_sayisi"] * 5
    )

    df["dijital_yuk"] = (
        df["uyku_oncesi_ekran_suresi_dk"] +
        np.log1p(df["uyku_oncesi_kafein_mg"])
    )

    df["fiziksel_saglik"] = (
        df["gunluk_adim_sayisi"] /
        df["vucut_kitle_indeksi"]
    )

    df["is_stres_yuku"] = (
        df["gunluk_calisma_saati"] *
        df["stres_skoru"]
    )

    df["rem_stres_denge"] = (
        df["rem_yuzdesi"] /
        (df["stres_skoru"] + 1)
    )

    df["yas_kare"] = df["yas"] ** 2

    return df

train = add_features(train)
test = add_features(test)

# =========================
# TARGET MEAN ENCODING
# =========================
high_signal_cols = [
    "meslek",
    "ruh_sagligi_durumu",
    "kronotip",
    "gun_tipi"
]

global_mean = train[TARGET].mean()

for col in high_signal_cols:
    mean_map = train.groupby(col)[TARGET].mean()

    train[f"{col}_mean_target"] = train[col].map(mean_map)
    test[f"{col}_mean_target"] = test[col].map(mean_map)

    train[f"{col}_mean_target"] = train[f"{col}_mean_target"].fillna(global_mean)
    test[f"{col}_mean_target"] = test[f"{col}_mean_target"].fillna(global_mean)

# =========================
# COMBO FEATURES
# =========================
train["meslek_ruh_combo"] = (
    train["meslek"].astype(str) + "_" +
    train["ruh_sagligi_durumu"].astype(str)
)

test["meslek_ruh_combo"] = (
    test["meslek"].astype(str) + "_" +
    test["ruh_sagligi_durumu"].astype(str)
)

combo_map = train.groupby("meslek_ruh_combo")[TARGET].mean()

train["meslek_ruh_combo_mean_target"] = (
    train["meslek_ruh_combo"].map(combo_map).fillna(global_mean)
)

test["meslek_ruh_combo_mean_target"] = (
    test["meslek_ruh_combo"].map(combo_map).fillna(global_mean)
)

train["ultra_combo"] = (
    train["meslek"].astype(str) + "_" +
    train["ruh_sagligi_durumu"].astype(str) + "_" +
    train["gun_tipi"].astype(str)
)

test["ultra_combo"] = (
    test["meslek"].astype(str) + "_" +
    test["ruh_sagligi_durumu"].astype(str) + "_" +
    test["gun_tipi"].astype(str)
)

ultra_map = train.groupby("ultra_combo")[TARGET].mean()

train["ultra_combo_mean_target"] = (
    train["ultra_combo"].map(ultra_map).fillna(global_mean)
)

test["ultra_combo_mean_target"] = (
    test["ultra_combo"].map(ultra_map).fillna(global_mean)
)

# =========================
# MODEL DATA
# =========================
X = train.drop(columns=[TARGET, "id"], errors="ignore")
y = train[TARGET]
X_test = test.drop(columns=["id"], errors="ignore")

cat_cols = X.select_dtypes(include="object").columns.tolist()

print("Final train shape:", X.shape)
print("Final test shape:", X_test.shape)
print("Categorical columns:", cat_cols)

# =========================
# SEED ENSEMBLE CONFIG
# =========================
seeds = [42, 2024, 2025, 3407, 777]

all_test_preds = np.zeros(len(X_test))
seed_results = []

# =========================
# TRAIN EACH SEED
# =========================
for seed in seeds:
    print("\n\n==============================")
    print(f"Training seed: {seed}")
    print("==============================")

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n========== Seed {seed} | Fold {fold} ==========")

        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]

        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = CatBoostRegressor(
            iterations=1200,
            depth=8,
            learning_rate=0.03,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed,
            verbose=300
        )

        model.fit(
            X_train_fold,
            y_train_fold,
            cat_features=cat_cols,
            eval_set=(X_val_fold, y_val_fold),
            use_best_model=True
        )

        val_preds = np.clip(model.predict(X_val_fold), 0, 10)
        oof[val_idx] = val_preds

        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
        fold_scores.append(fold_rmse)

        print(f"Seed {seed} Fold {fold} RMSE: {fold_rmse:.5f}")

        test_preds += model.predict(X_test) / kf.n_splits

    seed_oof_rmse = np.sqrt(mean_squared_error(y, oof))
    seed_mean_rmse = np.mean(fold_scores)

    print(f"\nSeed {seed} Mean RMSE: {seed_mean_rmse:.5f}")
    print(f"Seed {seed} OOF RMSE: {seed_oof_rmse:.5f}")

    seed_results.append({
        "seed": seed,
        "mean_rmse": seed_mean_rmse,
        "oof_rmse": seed_oof_rmse
    })

    test_preds = np.clip(test_preds, 0, 10)

    seed_submission = pd.DataFrame({
        "id": test["id"],
        TARGET: test_preds
    })

    seed_path = f"{SUBMISSION_DIR}/catboost_seed_{seed}_submission.csv"
    seed_submission.to_csv(seed_path, index=False)

    print(f"Saved seed submission: {seed_path}")

    all_test_preds += test_preds / len(seeds)

# =========================
# SAVE SEED RESULTS
# =========================
results_df = pd.DataFrame(seed_results).sort_values("oof_rmse")
results_path = f"{SUBMISSION_DIR}/seed_blend_results.csv"
results_df.to_csv(results_path, index=False)

print("\n========== SEED RESULTS ==========")
print(results_df)

# =========================
# SAVE BLENDED SUBMISSION
# =========================
all_test_preds = np.clip(all_test_preds, 0, 10)

blend_submission = pd.DataFrame({
    "id": test["id"],
    TARGET: all_test_preds
})

blend_path = f"{SUBMISSION_DIR}/catboost_seed_blend_submission.csv"
blend_submission.to_csv(blend_path, index=False)

print(f"\nSeed blend submission saved: {blend_path}")
print(blend_submission.head())
print(blend_submission.shape)