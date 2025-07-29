import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import KMeansSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import ClusterCentroids, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from typing import Union, Tuple



def load_data(path: str, has_labels: bool = True):
    """
    Load dataset from a CSV file.

    Args:
        path (str): Path to the dataset CSV file.
        has_labels (bool): Whether the dataset contains labels (target column 'Class').

    Returns:
        - If has_labels: (X, y)
        - If not: X
    """
    data = pd.read_csv(path)

    if has_labels:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return X, y
    else:
        return data
    
def apply_scaling(X, scaler_type='standard', scaling_on='train', fitted_scaler=None) -> Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler, RobustScaler]]:
    """
    Apply scaling to features.

    Args:
        X (DataFrame): Feature matrix.
        scaler_type (str): 'standard', 'minmax', or 'robust'.
        scaling_on (str): 'train', 'val', or 'test'.

    Returns:
        Tuple: (scaled_X, scaler)
    """
    if scaling_on == 'train':
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported scaler_type. Choose from: 'standard', 'minmax', 'robust'.")
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    elif scaling_on in ('val', 'test'):
        if fitted_scaler is None:
            raise ValueError("fitted_scaler must be provided for validation or test data.")
        X_scaled = fitted_scaler.transform(X)
        return X_scaled, fitted_scaler

    else:
        raise ValueError("scaling_on must be one of: 'train', 'val', or 'test'.")

def solve_imbalance(X, y, method: str):
    """
    Handle class imbalance using various resampling techniques.

    Args:
        X (DataFrame): Feature matrix.
        y (Series): Target labels.
        method (str): Resampling technique. Choose from:
            - 'kms'  : KMeansSMOTE
            - 'ros'  : RandomOverSampler
            - 'cc'   : ClusterCentroids
            - 'oss'  : OneSidedSelection
            - 'smote_enn' : SMOTE + Edited Nearest Neighbors
            - 'smote_tomek': SMOTE + Tomek Links
            - 'SMOTE': SMOTE(random_state=42, sampling_strategy=0.7)

    Returns:
        Tuple: (X_resampled, y_resampled)
    
    Raises:
        ValueError: If an invalid method is provided.
    """
    method = method.lower()  # Normalize to lowercase for user convenience
    
    resampling_methods = {
    # Oversampling Methods
    "ros": RandomOverSampler(sampling_strategy='auto', random_state=41),
    "SMOTE": SMOTE(sampling_strategy='minority', random_state=42),
    "kms": KMeansSMOTE(sampling_strategy='auto', random_state=41, k_neighbors=5, cluster_balance_threshold='auto'),
    
    # Undersampling Methods
    "cc": ClusterCentroids(sampling_strategy='auto', random_state=41, voting="auto"),
    "oss": OneSidedSelection(sampling_strategy='auto', random_state=41, n_neighbors=3),
    
    # Hybrid Methods (Over + Under Sampling)
    "smote_enn": SMOTEENN(sampling_strategy='auto', random_state=41),
    "smote_tomek": SMOTETomek(sampling_strategy='auto', random_state=41),
}

    if method not in resampling_methods:
        raise ValueError(f"Invalid method: {method}. Choose from {list(resampling_methods.keys())}")

    sampler = resampling_methods[method]
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled


print("data_utils.py ran successfully!")
