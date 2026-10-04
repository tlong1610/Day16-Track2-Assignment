import json
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else "~/ml-benchmark/creditcard.csv")
OUT_PATH = "benchmark_result.json"
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated")

# 1. Load data
t0 = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time = time.perf_counter() - t0

X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# 2. Train (no class weighting: keeps probabilities calibrated, threshold tuned below)
model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.02, num_leaves=31,
    min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0,
    metric="auc", n_jobs=2, random_state=42, verbose=-1)
t0 = time.perf_counter()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
train_time = time.perf_counter() - t0

# 3. Pick the decision threshold that maximizes F1 on validation, then evaluate on test
val_proba = model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.05, 0.95, 91)
threshold = float(max(thresholds,
                      key=lambda t: f1_score(y_val, (val_proba >= t).astype(int))))
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)

# 4. Inference latency (1 row) and throughput (1000 rows)
for i in range(10):
    model.predict(X_test.iloc[[i]])
lat = []
for i in range(100):
    t0 = time.perf_counter()
    model.predict(X_test.iloc[[i]])
    lat.append(time.perf_counter() - t0)

batch = X_test.iloc[:1000]
t0 = time.perf_counter()
model.predict(batch)
batch_time = time.perf_counter() - t0

result = {
    "load_time_s": round(load_time, 3),
    "train_time_s": round(train_time, 3),
    "best_iteration": int(model.best_iteration_),
    "decision_threshold": round(threshold, 2),
    "auc_roc": round(roc_auc_score(y_test, proba), 4),
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "f1_score": round(f1_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "inference_latency_1row_ms": round(float(np.mean(lat)) * 1000, 3),
    "inference_latency_1row_p95_ms": round(float(np.percentile(lat, 95)) * 1000, 3),
    "inference_1000rows_ms": round(batch_time * 1000, 3),
    "inference_throughput_rows_per_s": round(1000 / batch_time, 1),
}

for k, v in result.items():
    print(f"{k:35s} {v}")
with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {OUT_PATH}")
