# <span style="color:#FFA500;">💳 Credit Card Fraud Detection with MLflow</span>

A <span style="color:#00BFFF;">machine learning</span> project for identifying fraudulent credit card transactions. It includes full <span style="color:#32CD32;">data preprocessing</span>, <span style="color:#32CD32;">model training</span>, <span style="color:#32CD32;">evaluation</span>, and <span style="color:#32CD32;">experiment tracking</span> using **MLflow**.

---

## <span style="color:#32CD32;">🚀 Key Highlights:</span>  
- **Data Handling:** Complete preprocessing including scaling and imbalance correction.  
- **Modeling:** Logistic Regression, Random Forest, and MLP Classifier.  
- **Experiment Tracking:** Automated logging of parameters, metrics, and artifacts using <span style="color:#00CED1;">MLflow</span>.  
- **Evaluation:** Uses advanced metrics like <span style="color:#DC143C;">F1 Score</span> and <span style="color:#DC143C;">PR-AUC</span> to handle imbalanced data.

---

## <span style="color:#1E90FF;">📊 Model Performance:</span>  

| Model                   | F1-Train Score | F1-Test Score |
|------------------------|----------------|---------------|
| Logistic Regression     | 81.6%          | 77.72%        |
| Random Forest Classifier| 94.23%         | 82.13%        |
| MLP Classifier          | 97.71%         | 85.71%        |

> ⚠️ <span style="color:#DC143C;">Note:</span> Accuracy can be misleading in imbalanced datasets — use F1 Score and PR-AUC instead.

---

## <span style="color:#FF69B4;">🛠 Project Structure:</span>  
```
Credit Card Fraud Detection With Mlflow/
├── Data/
│   └── Compressed Data.zip
├── EDA/
│   └── EDA.ipynb
├── mlruns/                      # MLflow tracking files
├── Model/
│   ├── Data_processing.py
│   ├── modeling.py
│   ├── evaluate.py
│   ├── Train.py
│   ├── Test.py
│   └── Mlflow.py
├── Results/                    # Evaluation images and reports
├── Saved Model/
│   ├── Logistic Regression.pkl
│   ├── MLP Classifier.pkl
│   └── Random Forest Classifier.pkl
├── Terminal Screenshoot/
└── Credit Card Fraud Detection Report/
    └── Credit Card Fraud Detection Report.pdf
```

---

## <span style="color:#20B2AA;">⚙️ How to Run:</span>

1. <span style="color:#00CED1;">Clone the repository</span>  
```bash
git clone https://github.com/your-username/credit-card-fraud-detection-mlflow.git
cd credit-card-fraud-detection-mlflow
```

2. <span style="color:#00CED1;">Install dependencies</span>  
```bash
pip install -r requirements.txt
```

3. <span style="color:#00CED1;">Train the model</span>  
```bash
python Model/Train.py
```

4. <span style="color:#00CED1;">Run testing and track with MLflow</span>  
```bash
python Model/Test.py
```

5. <span style="color:#00CED1;">Launch MLflow UI</span>  
```bash
mlflow ui
```

---

## <span style="color:#DAA520;">📁 Output Details:</span>
- 📈 All evaluation images and classification reports are stored in `Results/`.
- 💾 Trained models saved with `joblib` in `Saved Model/`.
- 🧪 All experiments are tracked in the `mlruns/` directory.

---

## <span style="color:#8A2BE2;">📚 Technologies Used:</span>
- Python  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- MLflow  

---

## <span style="color:#FF4500;">👤 Author:</span>  
**Ahmed Reda Ahmed**
