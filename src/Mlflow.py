import os
import json
import logging
import warnings
import mlflow

from save_load_model import load_model
from evaluate_models import evaluate_model_custom_threshold, print_evaluation_metrics
from data_utils import load_data

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(format="🔹 [%(levelname)s] %(message)s", level=logging.INFO)

# === Model Config ===
model_name = "decision_tree"  
model_dir = "models"
model_path = os.path.join(model_dir, f"{model_name}.pkl")
threshold_path = os.path.join(
    "results", 
    model_name.lower(), 
    "best_threshold.json"
)

# === Result Directories ===
result_dir = os.path.join("results", model_name)
reports_dir = os.path.join(result_dir, "reports")
imgs_dir = os.path.join(result_dir, "imgs")
report_path = os.path.join(reports_dir, "test_classification_report.txt")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

# === Load Test Data ===
logging.info("📥 Loading test dataset...")
X_test, y_test = load_data("data/test.csv", has_labels=True)

# === Load Model ===
if not os.path.exists(model_path):
    logging.error(f"❌ Error: File '{model_path}' not found.")
    exit(1)

logging.info(f"📦 Loading model from {model_path}...")
model = load_model(model_path)
print(f"Model loaded from {model_path}")

# === Load Best Threshold ===
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        best_threshold = float(json.load(f)["best_threshold"])
    logging.info(f"🎯 Loaded best threshold: {best_threshold:.4f}")
else:
    best_threshold = 0.5
    logging.warning("⚠️ Threshold not found. Using default 0.5.")
    

# === Evaluate Model (Correct Unpacking) ===
logging.info("📊 Evaluating model on test set...")
test_metrics, _ = evaluate_model_custom_threshold(
    model, X_test, y_test, use_fixed_threshold=best_threshold
)

# === Save classification report ===
print_evaluation_metrics(
    test_metrics,
    title="(Test Set)",
    model_name=model_name,
    save_to_txt=True,
    txt_filename="test_classification_report.txt",
    save_dir=reports_dir
)

# === MLflow Tracking ===
with mlflow.start_run(run_name=f"{model_name}_test_run"):
    mlflow.set_tag("stage", "testing")
    mlflow.set_tag("model", model_name)

    # Log parameters
    mlflow.log_param("best_threshold", best_threshold)

    # Log metrics
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(f"test_{metric_name}", metric_value)

    # Log artifacts
    if os.path.exists(report_path):
        mlflow.log_artifact(report_path, artifact_path="reports")

    for img_file in os.listdir(imgs_dir):
        if img_file.endswith(".png"):
            mlflow.log_artifact(os.path.join(imgs_dir, img_file), artifact_path="imgs")

logging.info("✅ MLflow logging completed.")
