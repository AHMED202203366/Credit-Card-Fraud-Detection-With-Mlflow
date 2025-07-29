import os
import json
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

from data_utils import load_data, apply_scaling, solve_imbalance
from save_load_model import save_model
from models import select_model
from evaluate_models import evaluate_model_custom_threshold, print_evaluation_metrics
from plot_save_imgs import plot_and_save_metrics

def train_model(data_path, model_name, scaler_type='standard', imbalance_method=None):
    print("📥 Loading training dataset...")
    X_train, y_train = load_data(data_path)

    if imbalance_method:
        print(f"⚖️ Handling class imbalance using {imbalance_method} method...")
        # ✅ Apply sampling before scaling
        X_train_resampled, y_train_resampled = solve_imbalance(X_train, y_train, imbalance_method)

        print("📊 Scaling features in the resampled training dataset...")
        X_train_scaled, scaler = apply_scaling(X_train_resampled, scaler_type)

        print(f"🧠 Training the {model_name} model with imbalance handling...")
        model = select_model(X_train_scaled, y_train_resampled, model_name, use_class_weight=True)
        return model, X_train_scaled, y_train_resampled, scaler
    else:
        print("📊 Scaling features in the training dataset...")
        X_train_scaled, scaler = apply_scaling(X_train, scaler_type)

        print(f"🧠 Training the {model_name} model without imbalance handling...")
        model = select_model(X_train_scaled, y_train, model_name)
        return model, X_train_scaled, y_train, scaler


def save_model_info(model, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_info_path = os.path.join(save_dir, "model_info.txt")

    with open(model_info_path, "w", encoding="utf-8") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write("Used Parameters:\n")
        if hasattr(model, "get_params"):
            for k, v in model.get_params().items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  No parameter info available.\n")


if __name__ == "__main__":
    # === Configuration ===
    data_path = {
        'train': 'data/train.csv',
        'val': 'data/val.csv'
    }

    model_name = 'decision_tree'       # 'logistic_Regression', 'random_forest', 'svm', 'decision_tree', 'knn', 'naive_bayes', 'gbt', 'xgboost', 'mlp'
    imbalance_method = 'smote_tomek'   # optional: "ros", "SMOTE", etc.
    scaler = 'standard'                # optional: "minmax", "robust", etc.

    model_id = model_name.lower().replace(" ", "_")
    model_path = os.path.join("models", f"{model_id}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # === Train ===
    model, X_train, y_train, scaler_object = train_model(
        data_path=data_path['train'],
        model_name=model_name,
        scaler_type=scaler

    )
    save_model(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    # === Prepare results folders ===
    root_result_dir = os.path.join("results", model_id)
    imgs_dir = os.path.join(root_result_dir, "imgs")
    reports_dir = os.path.join(root_result_dir, "reports")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # === Train Evaluation ===
    print("📈 Evaluating on training data...")
    train_metrics, best_threshold = evaluate_model_custom_threshold(model, X_train, y_train)
    print_evaluation_metrics(
    train_metrics,
    title="Training Metrics",
    model_name=model_name,
    save_to_txt=True,
    txt_filename="train_classification_report.txt",  
    save_dir=reports_dir
  )

    plot_and_save_metrics(train_metrics, model_name=f"{model_name}_train", save_dir=imgs_dir)

    # === Validation Evaluation ===
    print("📥 Loading validation dataset...")
    X_val, y_val = load_data(data_path['val'])
    X_val_scaled, _ = apply_scaling(X_val, scaler, scaling_on='val', fitted_scaler=scaler_object)

    print("📉 Evaluating on validation set...")
    val_metrics, _ = evaluate_model_custom_threshold(model, X_val_scaled, y_val, use_fixed_threshold=float(best_threshold))
    print_evaluation_metrics(
    val_metrics,
    title="Validation Metrics",
    model_name=model_name,
    save_to_txt=True,
    txt_filename="val_classification_report.txt",  
    save_dir=reports_dir
    )
    plot_and_save_metrics(val_metrics, model_name=f"{model_name}_val", save_dir=imgs_dir)

    # === Save Threshold ===
    threshold_path = os.path.join(root_result_dir, "best_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"best_threshold": float(best_threshold)}, f)
    print(f"📎 Best threshold saved to {threshold_path} (value = {best_threshold:.4f})")

    # === Save Model Info ===
    save_model_info(model, model_name, root_result_dir)

    print("🏁 All tasks completed successfully!")
