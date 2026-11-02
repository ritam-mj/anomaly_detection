"""
test_mock_data.py
-----------------
Automated testing and benchmarking script for the Isolation Forest Anomaly Detection Model.
Generates realistic mock supply chain telemetry data across normal operations and 
various acute risk scenarios, then evaluates model accuracy, precision, recall, F1, 
ROC-AUC, per-scenario detection rate, and edge inference latency.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Optional imports based on backend
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


DEFAULT_FEATURES = [
    "temperature",
    "humidity",
    "vibration_level",
    "stock_quantity",
    "supplier_rating"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Isolation Forest on synthetic mock data.")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Total number of mock telemetry packets to generate (default: 1000)"
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.20,
        help="Proportion of synthetic anomalies (default: 0.20 = 20%%)"
    )
    parser.add_argument(
        "--model-type",
        choices=["onnx", "joblib", "auto"],
        default="auto",
        help="Model runtime backend to use (default: auto)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (.onnx or .joblib)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom anomaly score threshold (default: None, uses model's native 0.0 boundary)"
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional file path to save the generated mock data and predictions (e.g. data/mock_test_results.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible mock generation (default: 42)"
    )
    return parser.parse_args()


def load_feature_metadata(base_dir=None):
    """Loads feature definitions and baseline statistics if available."""
    candidates = [
        "model_features.json",
        "detection_model/model_features.json",
        os.path.join(base_dir or "", "model_features.json"),
        os.path.join(base_dir or "", "detection_model", "model_features.json"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                with open(c, "r") as f:
                    meta = json.load(f)
                    print(f"Loaded feature metadata from: {c}")
                    return meta.get("features", DEFAULT_FEATURES), meta.get("feature_stats", {})
            except Exception:
                pass
    return DEFAULT_FEATURES, {}


def generate_mock_telemetry(n_samples: int = 1000, anomaly_ratio: float = 0.20, seed: int = 42):
    """
    Generates a realistic mock dataset containing nominal operations and
    simulated risk scenarios (thermal breakdown, cargo impact shock, humidity ingress,
    critical stock depletion, multi-sensor compound shocks, and supplier risk drift).
    """
    np.random.seed(seed)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    records = []

    # 1. Normal Operations (Label: 0)
    for _ in range(n_normal):
        records.append({
            "scenario": "Nominal Transit",
            "temperature": float(np.random.normal(24.0, 2.8)),       # 18 - 30 C
            "humidity": float(np.clip(np.random.normal(50.0, 6.0), 20.0, 75.0)), # 35 - 65%
            "vibration_level": float(np.clip(np.random.exponential(0.8) + 0.2, 0.05, 2.2)), # Smooth transit
            "stock_quantity": int(np.clip(np.random.normal(280, 75), 80, 500)),
            "supplier_rating": float(np.clip(np.random.normal(3.8, 0.6), 2.5, 5.0)),
            "is_anomaly": 0
        })

    # 2. Simulated Risk / Anomaly Scenarios (Label: 1)
    # Distribute anomalies across distinct realistic supply chain failure modes
    scenario_distribution = [
        ("Thermal Shock (Heatwave/Refrigeration Loss)", 0.25),
        ("Extreme Vibration (Impact/Cargo Drop)", 0.25),
        ("Moisture Ingress (Severe Condensation/Water)", 0.15),
        ("Critical Inventory Collapse (Stockout Risk)", 0.15),
        ("Compound Multi-Sensor Failure (Heat + Shock)", 0.10),
        ("High-Risk Supplier Telemetry Drift", 0.10),
    ]

    for scenario_name, weight in scenario_distribution:
        count = int(n_anomalies * weight)
        for _ in range(count):
            temp = float(np.random.normal(24.0, 2.5))
            hum = float(np.random.normal(50.0, 5.0))
            vib = float(np.random.normal(1.0, 0.3))
            stock = int(np.random.normal(280, 60))
            rating = float(np.random.normal(3.8, 0.5))

            if scenario_name == "Thermal Shock (Heatwave/Refrigeration Loss)":
                temp = float(np.random.choice([
                    np.random.uniform(43.0, 58.0),
                    np.random.uniform(-12.0, 2.0)
                ]))
            elif scenario_name == "Extreme Vibration (Impact/Cargo Drop)":
                vib = float(np.random.uniform(4.5, 9.5))
            elif scenario_name == "Moisture Ingress (Severe Condensation/Water)":
                hum = float(np.random.uniform(86.0, 99.0))
            elif scenario_name == "Critical Inventory Collapse (Stockout Risk)":
                stock = int(np.random.uniform(0, 15))
            elif scenario_name == "Compound Multi-Sensor Failure (Heat + Shock)":
                temp = float(np.random.uniform(42.0, 55.0))
                hum = float(np.random.uniform(80.0, 95.0))
                vib = float(np.random.uniform(4.0, 8.0))
            elif scenario_name == "High-Risk Supplier Telemetry Drift":
                rating = float(np.random.uniform(1.0, 1.6))
                vib = float(np.random.uniform(2.8, 4.2))
                temp = float(np.random.uniform(33.0, 42.0))

            records.append({
                "scenario": scenario_name,
                "temperature": round(temp, 2),
                "humidity": round(hum, 2),
                "vibration_level": round(vib, 2),
                "stock_quantity": int(max(0, stock)),
                "supplier_rating": round(float(np.clip(rating, 1.0, 5.0)), 2),
                "is_anomaly": 1
            })

    # Handle any remaining rounding difference
    while len(records) < n_samples:
        records.append({
            "scenario": "Thermal Shock (Heatwave/Refrigeration Loss)",
            "temperature": float(np.random.uniform(45.0, 55.0)),
            "humidity": float(np.random.normal(50.0, 5.0)),
            "vibration_level": float(np.random.normal(1.0, 0.3)),
            "stock_quantity": int(np.random.normal(250, 50)),
            "supplier_rating": float(np.random.normal(3.5, 0.5)),
            "is_anomaly": 1
        })

    df_mock = pd.DataFrame(records)
    # Shuffle dataset
    df_mock = df_mock.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_mock


class ModelInferenceEngine:
    """Wrapper that abstracts inference over both ONNX Runtime and Scikit-Learn pipelines."""

    def __init__(self, model_type="auto", model_path=None):
        self.model_type = model_type
        self.model_path = model_path
        self.session = None
        self.pipeline = None
        self._init_model()

    def _init_model(self):
        candidates_onnx = [
            self.model_path if (self.model_path and self.model_path.endswith(".onnx")) else None,
            "supply_chain_isolation_forest.onnx",
            "detection_model/supply_chain_isolation_forest.onnx",
        ]
        candidates_joblib = [
            self.model_path if (self.model_path and self.model_path.endswith(".joblib")) else None,
            "supply_chain_isolation_forest.joblib",
            "detection_model/supply_chain_isolation_forest.joblib",
        ]

        found_onnx = next((c for c in candidates_onnx if c and os.path.exists(c)), None)
        found_joblib = next((c for c in candidates_joblib if c and os.path.exists(c)), None)

        if self.model_type in ["onnx", "auto"] and found_onnx and HAS_ORT:
            self.model_type = "onnx"
            self.model_path = found_onnx
            self.session = ort.InferenceSession(found_onnx, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            print(f"Loaded ONNX Model from: {found_onnx}")
            return

        if self.model_type in ["joblib", "auto"] and found_joblib and HAS_JOBLIB:
            self.model_type = "joblib"
            self.model_path = found_joblib
            self.pipeline = joblib.load(found_joblib)
            print(f"Loaded Scikit-Learn Joblib Model from: {found_joblib}")
            return

        raise FileNotFoundError(
            f"Could not load model. Looked for ONNX ({found_onnx}) and Joblib ({found_joblib}). "
            "Please ensure you run 'python detection_model/isolation_forest.py' first."
        )

    def predict_batch(self, X: np.ndarray):
        """
        Runs batch prediction.
        Returns:
            binary_preds: np.ndarray (0 = Normal, 1 = Anomaly)
            scores: np.ndarray (Continuous anomaly score; negative = anomaly)
        """
        X_float = X.astype(np.float32)
        if self.model_type == "onnx":
            outputs = self.session.run(None, {self.input_name: X_float})
            raw_labels = outputs[0].flatten()  # 1 = normal, -1 = anomaly
            scores = outputs[1].flatten()
            # Map 1 -> 0 (Normal) and -1 -> 1 (Anomaly)
            binary_preds = np.where(raw_labels == -1, 1, 0)
            return binary_preds, scores
        else:
            raw_preds = self.pipeline.predict(X_float)
            scores = self.pipeline.decision_function(X_float)
            binary_preds = np.where(raw_preds == -1, 1, 0)
            return binary_preds, scores

    def benchmark_latency(self, X_sample: np.ndarray, n_iterations: int = 500):
        """Measures single-packet edge inference latency."""
        sample_packet = X_sample[0:1].astype(np.float32)
        latencies = []

        # Warm-up run
        for _ in range(20):
            if self.model_type == "onnx":
                _ = self.session.run(None, {self.input_name: sample_packet})
            else:
                _ = self.pipeline.predict(sample_packet)

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            if self.model_type == "onnx":
                _ = self.session.run(None, {self.input_name: sample_packet})
            else:
                _ = self.pipeline.predict(sample_packet)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

        latencies = np.array(latencies)
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "throughput_ips": float(1000.0 / np.mean(latencies))
        }


def print_separator(char="=", length=72):
    print(char * length)


def evaluate_mock_performance():
    args = parse_args()
    print_separator()
    print("   SUPPLY CHAIN ISOLATION FOREST - MOCK PERFORMANCE EVALUATION")
    print_separator()

    # Load feature schema
    feature_cols, _ = load_feature_metadata()
    print(f"Target Features ({len(feature_cols)}): {feature_cols}")

    # Generate mock data
    print(f"\nGenerating {args.samples} mock telemetry samples (Anomaly ratio: {args.anomaly_ratio*100:.1f}%)...")
    df_mock = generate_mock_telemetry(
        n_samples=args.samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed
    )

    y_true = df_mock["is_anomaly"].values
    X_test = df_mock[feature_cols].values

    # Load model
    model = ModelInferenceEngine(model_type=args.model_type, model_path=args.model_path)

    # Run predictions
    print("\nRunning inference over mock test dataset...")
    y_pred, decision_scores = model.predict_batch(X_test)
    if args.threshold is not None:
        print(f"Applying custom decision threshold: {args.threshold} (anomaly if score < {args.threshold})")
        y_pred = np.where(decision_scores < args.threshold, 1, 0)

    df_mock["predicted_label"] = y_pred
    df_mock["anomaly_score"] = np.round(decision_scores, 4)

    # 1. Overall Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, -decision_scores)
    except Exception:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*72)
    print(" 1. GLOBAL PERFORMANCE METRICS")
    print("="*72)
    print(f"  * Accuracy:        {acc * 100:.2f}%")
    print(f"  * Precision:       {prec * 100:.2f}%  (When model alerts, how often is it real?)")
    print(f"  * Recall:          {rec * 100:.2f}%  (Of all anomalies, how many were caught?)")
    print(f"  * F1-Score:        {f1:.4f}")
    print(f"  * ROC-AUC Score:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"  +---------------------+-------------------+-------------------+")
    print(f"  |                     | Pred Normal (0)   | Pred Anomaly (1)  |")
    print(f"  +---------------------+-------------------+-------------------+")
    print(f"  | True Normal (0)     | TN: {tn:<13} | FP: {fp:<13} |")
    print(f"  | True Anomaly (1)    | FN: {fn:<13} | TP: {tp:<13} |")
    print(f"  +---------------------+-------------------+-------------------+")

    # 2. Per-Scenario Detection Breakdown
    print("\n" + "="*72)
    print(" 2. PER-SCENARIO DETECTION BREAKDOWN")
    print("="*72)
    scenario_groups = df_mock.groupby("scenario")
    print(f"  {'Scenario Name':<45} | {'Count':<5} | {'Detected':<8} | {'Rate':<7} | {'Avg Score':<9}")
    print("  " + "-"*82)

    for scenario_name, group in scenario_groups:
        total = len(group)
        is_true_anom = group["is_anomaly"].iloc[0] == 1
        caught = (group["predicted_label"] == 1).sum() if is_true_anom else (group["predicted_label"] == 0).sum()
        rate = (caught / total) * 100
        avg_score = group["anomaly_score"].mean()

        status_sym = "[OK]  " if rate >= 80 else ("[WARN]" if rate >= 50 else "[FAIL]")
        print(f"  {status_sym} {scenario_name:<39} | {total:<5} | {caught:<8} | {rate:>5.1f}% | {avg_score:>+8.4f}")

    # 3. Latency Benchmarking
    print("\n" + "="*72)
    print(" 3. EDGE INFERENCE LATENCY BENCHMARK")
    print("="*72)
    latency_stats = model.benchmark_latency(X_test, n_iterations=300)
    print(f"  * Backend Engine:     {model.model_type.upper()}")
    print(f"  * Mean Latency:       {latency_stats['mean_ms']:.4f} ms ({latency_stats['mean_ms']*1000:.1f} us)")
    print(f"  * Median Latency:     {latency_stats['median_ms']:.4f} ms")
    print(f"  * 95th Percentile:    {latency_stats['p95_ms']:.4f} ms")
    print(f"  * 99th Percentile:    {latency_stats['p99_ms']:.4f} ms")
    print(f"  * Edge Throughput:    ~{latency_stats['throughput_ips']:,.0f} packets/second")

    # 4. Optional CSV Export
    if args.save_csv:
        save_path = os.path.abspath(args.save_csv)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df_mock.to_csv(save_path, index=False)
        print(f"\nSaved synthetic test dataset & predictions to: '{save_path}'")

    print_separator()
    print("   EVALUATION COMPLETE")
    print_separator()


if __name__ == "__main__":
    evaluate_mock_performance()
