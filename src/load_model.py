import os
import logging
import warnings
import json

from save_load_model import load_model
from evaluate_models import evaluate_model_custom_threshold, print_evaluation_metrics, evaluate_model_default_threshold
from plot_save_imgs import plot_and_save_metrics
from data_utils import load_data

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(format="🔹 [%(levelname)s] %(message)s", level=logging.INFO)

# === Define model details ===
model_name = "decision_tree"  
model_dir = "models"
model_path = os.path.join(model_dir, f"{model_name.lower()}.pkl")
threshold_path = os.path.join(
    "results", 
    model_name.lower(), 
    "best_threshold.json"
)

# === Load test data ===
logging.info("📥 Loading test dataset...")
X_test, y_test = load_data("data/test.csv", has_labels=True)

# === Load model ===
if not os.path.exists(model_path):
    logging.error(f"❌ Error: File '{model_path}' not found.")
    exit(1)

logging.info(f"📦 Loading model from {model_path}...")
model = load_model(model_path)

# === Load best threshold ===
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        best_threshold = float(json.load(f)["best_threshold"])
    logging.info(f"🎯 Loaded best threshold: {best_threshold:.4f}")
else:
    best_threshold = 0.5
    logging.warning("⚠️ Best threshold not found. Using default 0.5.")   

# === Evaluate the model ===
logging.info("📊 Evaluating model on test set...")
test_metrics, _= evaluate_model_custom_threshold(model, X_test, y_test)

# === Setup result directories ===
result_dir = os.path.join("results", model_name.lower())
reports_dir = os.path.join(result_dir, "reports")
imgs_dir = os.path.join(result_dir, "imgs")
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(imgs_dir, exist_ok=True)

# === Save and print evaluation metrics ===
print("\n=== Model Evaluation Summary ===")
print_evaluation_metrics(
    test_metrics,
    model_name=model_name,
    title="Testing Metrics",
    save_to_txt=True,
    txt_filename="test_classification_report.txt",
    save_dir=reports_dir,  # ✅ Save text report in reports/
)

# === Plot and save visualizations ===
plot_and_save_metrics(test_metrics, model_name=f"{model_name}_test", save_dir=imgs_dir)  # ✅ Save plots in imgs/

logging.info("✅ Testing and reporting completed.")

