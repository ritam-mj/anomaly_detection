import os
import sys
import json
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description="Train Isolation Forest pipeline on telemetry dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to supply chain risk dataset CSV (default: auto-detected in data/)"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomaly contamination fraction (default: 0.05)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save exported model artifacts (default: current working directory)"
    )
    return parser.parse_args()

def locate_dataset(dataset_arg=None):
    if dataset_arg and os.path.exists(dataset_arg):
        return os.path.abspath(dataset_arg)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    candidate_paths = [
        os.path.join(project_root, "data", "supply_chain_risk_dataset.csv"),
        os.path.join("data", "supply_chain_risk_dataset.csv"),
        os.path.join(script_dir, "data", "supply_chain_risk_dataset.csv"),
        "supply_chain_risk_dataset.csv",
        *glob.glob("data/*.csv"),
        *glob.glob("*.csv"),
    ]

    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "Could not find dataset. Please provide path via --dataset (e.g. data/supply_chain_risk_dataset.csv)."
    )

args = parse_args()

# -------------------------------------------------------------
# 1. Dataset Loading & Feature Extraction
# -------------------------------------------------------------
dataset_path = locate_dataset(args.dataset)
print(f"Loading dataset: {dataset_path}")
df = pd.read_csv(dataset_path)

# Extract numerical columns (IoT readings, logistics delays, lead times, costs)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Identify non-predictive ID / index columns AND ground-truth labels (prevent target leakage)
def is_ignored_column(col_name: str) -> bool:
    lower = col_name.lower().strip()
    if lower in {"id", "index", "target", "label", "risk"}:
        return True
    if lower.startswith("unnamed"):
        return True
    if any(lower.endswith(s) for s in ["_id", "_index", "_label", "_target", "_risk"]):
        return True
    if any(lower.startswith(s) for s in ["id_", "index_"]):
        return True
    if "manual_risk" in lower:
        return True
    return False

feature_cols = [c for c in numerical_cols if not is_ignored_column(c)]

print(f"Selected {len(feature_cols)} telemetry feature columns: {feature_cols}")

# Impute missing values with column medians
medians = df[feature_cols].median().to_dict()
X_data = df[feature_cols].fillna(df[feature_cols].median()).values.astype(np.float32)

# -------------------------------------------------------------
# 2. Pipeline Definition: Scaler + Isolation Forest
# -------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=100,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1
    ))
])

print(f"Fitting Isolation Forest pipeline on baseline telemetry (contamination={args.contamination})...")
pipeline.fit(X_data)

# Baseline check: 1 = normal, -1 = anomaly
native_preds = pipeline.predict(X_data[:5])
native_scores = pipeline.decision_function(X_data[:5])
print(f"Native Sklearn sample predictions (1=Normal, -1=Anomaly): {native_preds}")
print(f"Native Sklearn sample decision scores: {np.round(native_scores, 4)}")

# -------------------------------------------------------------
# 3. Export to ONNX via skl2onnx & Save Metadata
# -------------------------------------------------------------
output_dir = args.output_dir if args.output_dir else os.getcwd()
os.makedirs(output_dir, exist_ok=True)

# Define input schema: dynamic batch dimension (None) and fixed feature length
num_features = X_data.shape[1]
initial_type = [("float_input", FloatTensorType([None, num_features]))]

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_type,
    target_opset={"": 14, "ai.onnx.ml": 3}
)

onnx_filename = os.path.join(output_dir, "supply_chain_isolation_forest.onnx")
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

joblib_filename = os.path.join(output_dir, "supply_chain_isolation_forest.joblib")
joblib.dump(pipeline, joblib_filename)

# Export feature metadata for inference and testing
feature_meta = {
    "features": feature_cols,
    "num_features": len(feature_cols),
    "medians": medians,
    "feature_stats": {
        col: {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }
        for col in feature_cols
    }
}
meta_filename = os.path.join(output_dir, "model_features.json")
with open(meta_filename, "w") as f:
    json.dump(feature_meta, f, indent=2)

print(f"Successfully exported ONNX pipeline to: '{onnx_filename}'")
print(f"Successfully exported Scikit-Learn pipeline to: '{joblib_filename}'")
print(f"Saved feature metadata to: '{meta_filename}'")

# -------------------------------------------------------------
# 4. Parity Verification with ONNX Runtime
# -------------------------------------------------------------
ort_session = ort.InferenceSession(onnx_filename, providers=["CPUExecutionProvider"])

input_name = ort_session.get_inputs()[0].name
output_names = [out.name for out in ort_session.get_outputs()]

# Query ONNX Runtime
ort_inputs = {input_name: X_data[:5]}
ort_outputs = ort_session.run(output_names, ort_inputs)

# skl2onnx exports:
# Output 0: Predicted labels (int64) -> 1 (normal) / -1 (anomaly)
# Output 1: Raw anomaly scores / path lengths
onnx_preds = ort_outputs[0].flatten()
onnx_scores = ort_outputs[1].flatten()

print(f"ONNX Runtime sample predictions: {onnx_preds}")
print(f"ONNX Runtime sample decision scores: {np.round(onnx_scores, 4)}")

# Confirm prediction alignment
np.testing.assert_array_equal(native_preds, onnx_preds, err_msg="ONNX and Scikit-Learn predictions do not match!")
print("Mathematical parity verified: ONNX output matches Scikit-Learn.")

# -------------------------------------------------------------
# 5. Real-Time Telemetry Streaming Simulation
# -------------------------------------------------------------
def process_realtime_packet(packet_features: np.ndarray, session: ort.InferenceSession):
    """
    Simulates edge validation: receives raw unscaled telemetry,
    runs the internal Scaler + Isolation Forest, and outputs an alert status.
    """
    payload = packet_features.reshape(1, -1).astype(np.float32)
    label, score = session.run(None, {session.get_inputs()[0].name: payload})
    
    is_anomaly = (label[0][0] == -1)
    anomaly_score = float(score[0][0])
    return is_anomaly, anomaly_score

print("\n--- Real-Time Telemetry Alert Evaluation ---")
# Pick a random sample and introduce an artificial shock anomaly
nominal_sample = X_data[0].copy()
shock_sample = X_data[0].copy()
shock_sample[0] = 55.0  # Simulate severe heatwave / cold-chain refrigeration failure (55 C)
shock_sample[2] = 5.5   # Simulate extreme vibration shock / vehicle impact (5.5 g)

for tag, sample in [("Nominal Transit", nominal_sample), ("Severe Telemetry Shock", shock_sample)]:
    alert, score = process_realtime_packet(sample, ort_session)
    status = "[ALERT] Anomaly Detected" if alert else "[OK] NORMAL"
    print(f"[{tag}] Score: {score:+.4f} | Status: {status}")