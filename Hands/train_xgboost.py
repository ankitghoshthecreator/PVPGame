"""
train_xgboost.py — Sign language landmark classifier training script using XGBoost.

Reads data/sign_data.csv, trains an XGBoost model
on the 63-dimensional normalized landmark features, evaluates it,
and saves the best-performing model + label encoder to models/.

Usage:
    python train_xgboost.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------
# CONFIGURABLE CONSTANTS
# ---------------------------------------------
DATA_CSV       = os.path.join("data", "sign_data.csv")
MODELS_DIR     = "models"
MODEL_PATH     = os.path.join(MODELS_DIR, "sign_model.joblib")
ENCODER_PATH   = os.path.join(MODELS_DIR, "label_encoder.joblib")

TEST_SIZE      = 0.20     # Fraction of data held out for final evaluation
RANDOM_STATE   = 42       # Seed for reproducibility

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH    = 3      # Keep shallow for speed and to prevent overfitting
XGB_LEARNING_RATE = 0.1
XGB_N_JOBS       = -1     # Use all CPU cores

# ── Cross-validation ──────────────────────────────────────────────────────────
CV_FOLDS = 5


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _divider(char="-", width=60):
    print(char * width)


def _print_eval(name: str, y_true, y_pred, class_names):
    """Print accuracy, per-class F1, and a text confusion matrix."""
    _divider()
    print(f"  Model: {name}")
    _divider()

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = true, cols = predicted):")
    # Pretty-print with class names as headers
    header = "         " + "  ".join(f"{c[:6]:>6}" for c in class_names)
    print(header)
    for row_label, row in zip(class_names, cm):
        print(f"  {row_label[:6]:>6}  " + "  ".join(f"{v:>6}" for v in row))
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(DATA_CSV):
        print(f"[ERROR] Data file not found: {DATA_CSV}")
        print("        Run collect_data.py first to generate training data.")
        sys.exit(1)

    print(f"[INFO] Loading data from {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)

    label_col  = "label"
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_col].values

    print(f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(set(y_raw))} classes -> {sorted(set(y_raw))}")

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    # ── 3. Train / test split (stratified) ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[INFO] Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ── 4. Define candidate model ─────────────────────────────────────────────
    # Depending on the number of classes, we use objective multi:softprob or binary:logistic
    objective = 'multi:softprob' if len(class_names) > 2 else 'binary:logistic'
    
    clf = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        objective=objective,
        n_jobs=XGB_N_JOBS,
        random_state=RANDOM_STATE,
    )
    
    # ── 5. Cross-validate, then evaluate on held-out test set ─────────────────
    name = "XGBoost"
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[INFO] Training {name} ...")

    # 5a. Cross-validation on training data (gives a robust accuracy estimate)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"       {CV_FOLDS}-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5b. Fit on full training set, then evaluate on test set
    clf.fit(X_train, y_train)
    y_pred    = clf.predict(X_test)
    test_acc  = (y_pred == y_test).mean()
    test_f1   = f1_score(y_test, y_pred, average="weighted")

    print(f"       Test accuracy: {test_acc:.4f}   Weighted F1: {test_f1:.4f}")
    _print_eval(name, y_test, y_pred, class_names)

    # ── 6. Save model + encoder ───────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENCODER_PATH)

    print(f"[INFO] Model  saved -> {os.path.abspath(MODEL_PATH)}")
    print(f"[INFO] Encoder saved -> {os.path.abspath(ENCODER_PATH)}")
    print("\n[DONE] Training complete. You can now use these artifacts in your inference script.")


if __name__ == "__main__":
    main()
