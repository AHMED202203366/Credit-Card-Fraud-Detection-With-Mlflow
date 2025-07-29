# <span style="color:#FFA500;">💳 Credit Card Fraud Detection with MLflow</span>

A <span style="color:#00BFFF;">machine learning</span> project for identifying fraudulent credit card transactions. It includes full <span style="color:#32CD32;">data preprocessing</span>, <span style="color:#32CD32;">model training</span>, <span style="color:#32CD32;">evaluation</span>, and <span style="color:#32CD32;">experiment tracking</span> using **MLflow**.

---

## <span style="color:#32CD32;">🚀 Key Highlights:</span>  
- **Data Handling:** Complete preprocessing including scaling and imbalance correction.  
- **Modeling:** Logistic Regression, Random Forest, XGBoost, and MLP Classifier.  
- **Experiment Tracking:** Automated logging of parameters, metrics, and artifacts using <span style="color:#00CED1;">MLflow</span>.  
- **Evaluation:** Uses advanced metrics like <span style="color:#DC143C;">F1 Score</span> and <span style="color:#DC143C;">PR-AUC</span> to handle imbalanced data.

---

## <span style="color:#1E90FF;">📊 Model Performance:</span>  

| Model                   | F1-Train Score | F1-Test Score |
|-------------------------|----------------|---------------|
| XGboost Classifier      | 1.0000         | 0.8200        |
| Random Forest Classifier| 0.8897         | 0.8351        |
| Decsion tree            | 0.8752         | 0.7614        |

> ⚠️ <span style="color:#DC143C;">Note:</span> Accuracy can be misleading in imbalanced datasets — use F1 Score and PR-AUC instead.

---

## <span style="color:#FF69B4;">🛠 Project Structure:</span>



<details>
<summary>📁 Click to expand</summary>

```bash
Credit Card Fraud Detection/
├── Data/                                 # Raw & processed datasets
│   └── compressed_data.zip
│
├── EDA/                                  # Exploratory Data Analysis
│   └── eda.ipynb
│
├── mlruns/                               # MLflow experiment tracking
│
├── models/                               # Saved trained models
│   ├── logistic_regression.pkl
│   ├── mlp.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── requirements/                         # Project dependencies
│   └── requirements.txt
│
├── results/                              # Output per model
│   ├── mlp/
│   │   ├── imgs/
│   │   └── reports/
├   |    ── best_threshold.json
│   │   └── model_info.txt
│   └── ... (any future models)
│   ├── random_forest/
│   │   ├── imgs/
│   │   └── reports/
         ── best_threshold.json
│   │   └── model_info.txt
│   └── ... (any future models)
│   ├── xgboost/
│   │   ├── imgs/
│   │   └── reports/
         ── best_threshold.json
│   │   └── model_info.txt
│   └── ... (any future models)
│   ├── decision_tree/
│   │   ├── imgs/
│   │   ├── reports/
│   │   ├── best_threshold.json
│   │   └── model_info.txt
│   └── ... (any future models)
│
├── src/                                  # Core codebase
│   ├── data_utils.py
│   ├── modeling.py
│   ├── evaluate_metrics.py
│   ├── train_ml.py
│   ├── train_nn.py
│   ├── train_all.py
│   ├── mlflow_runner.py
│   ├── save_load_models.py
│   └── plot_save_imgs.py


Key features of this README:
1. Uses proper Markdown syntax for code blocks
2. Includes clear step-by-step instructions
3. Formats all terminal commands in code blocks
4. Adds a dedicated section for MLflow UI access
5. Includes the URL for accessing MLflow (localhost:5000)
6. Uses bold headers for better readability

To use this:
1. Replace `<repository-url>` with your actual Git URL
2. Replace `<project-folder>` with your actual directory name
3. Save this content in a file named `README.md` in your project's root directory

The final result will display like this in GitHub/Markdown viewers:

---

# How to Run

Follow these steps to set up and run the project:

1. **Clone the repository** or navigate to the project folder:
   ```bash
   git clone <repository-url>
   cd <project-folder>

