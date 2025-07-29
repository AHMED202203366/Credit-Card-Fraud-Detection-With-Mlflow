# models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

def select_model(X_train, y_train, model_name='logistic', use_class_weight=False):
    """
    Select and configure a machine learning model with proper imbalance handling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model to use
        use_class_weight: Whether to apply class weighting (use False when applying external imbalance handling)
    
    Returns:
        Configured and trained model
    """
    # Calculate class ratio for imbalance handling
    class_ratio = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    
    # Create class weight dictionary if needed
    class_weight = {0: 1, 1: min(100, class_ratio)} if use_class_weight else None
    
    if model_name == 'logistic_Regression':
        model = LogisticRegression(
            max_iter=10000,
            class_weight=class_weight,
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        )
        
    elif model_name == 'random_forest':
        model = RandomForestClassifier(
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4
        )
        
    elif model_name == 'svm':
        model = SVC(
            class_weight=class_weight,
            probability=True,
            random_state=42,
            kernel='rbf',
            gamma='scale',
            C=1.0
        )
        
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=42,
            max_depth=8,
            min_samples_leaf=5
        )
        
    elif model_name == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=15,
            weights='distance',
            n_jobs=-1
        )
        
    elif model_name == 'naive_bayes':
        model = GaussianNB()
        
    elif model_name == 'gbt':
        model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8
        )
        
    elif model_name == 'xgboost':
        model = XGBClassifier(
            scale_pos_weight=class_ratio if use_class_weight else 1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        
    elif model_name == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=2000,
            solver='adam',
            activation='relu',
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            alpha=0.001,
            batch_size=256
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # Train the model
    model.fit(X_train, y_train)
    return model

print("✅ models.py ran successfully!")