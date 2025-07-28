Credit Card Fraud Detection
A machine learning project aimed at detecting fraudulent banking transactions using classification models. The project includes complete data preprocessing, training, evaluation, and logging using MLflow.

Problem Statement
Banking systems handle millions of transactions daily. This project aims to build a predictive system to identify fraudulent credit card transactions to help banks and financial institutions prevent losses.

Project Structure
Credit Card Fraud Detection With Mlflow/
├── Data/
│   ├── Compressed Data.zip
│   
│   
│   
│
├── EDA/
│   └── EDA.ipynb
│
├── mlruns/                      # MLflow tracking files
│
├── Model/
│   ├── Data_processing.py
│   ├── modeling.py
│   ├── evaluate.py
│   ├── Train.py
│   ├── Test.py
│   └── Mlflow.py
│
├── Results/
│   └── [Evaluation curve images and classification reports]
│
├── Saved Model/
│   ├── Logistic Regression.pkl
│   ├── MLP Classifier.pkl
│   └── Random Forest Classifier.pkl
│
├── Terminal Screenshoot/
│
│__Credit Card Fraud Detection Report/
   ├──Credit Card Fraud Detection Report.pdf

Models & Performance
Model 	F1-Train Accuracy	F1-Test Accuracy
Logistic Regression	81.6%	77.72%
Random Forest Classifier	94.23%	82.13%
MLP Classifier	97.71%	85.71%
⚠️ Note: Accuracy can be misleading in imbalanced datasets. Consider using F1 Score and PR-AUC for better evaluation.

 How to Run
Clone the repository or navigate to the project folder.
Install the required libraries:
pip install -r requirements.txt
Run training:
python Model/Train.py
Run testing and logging to MLflow:
python Model/Test.py
MLflow UI can be viewed by running:

mlflow ui
Technologies Used
Python
Scikit-learn
Pandas / NumPy
Matplotlib / Seaborn
MLflow
 Evaluation Metrics
Accuracy (Don't depend on it; it's misleading in imbalanced data)
F1 Score
ROC-AUC
PR-AUC
Confusion Matrix
Notes
All evaluation images and classification reports are saved inside the Results/ directory.
Models are saved using joblib inside Saved Model/.
Project tracks training and testing via MLflow for better experiment tracking.
Author
Ahmed Mohamed Hussain
