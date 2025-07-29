import pickle
import os

def save_model(model, filename):
    """
    Save a trained model to a .pkl file.

    Args:
        model: The trained model object to save.
        filename (str): Path to the .pkl file (e.g., 'model_saver/model.pkl').
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename):
    """
    Load a model from a .pkl file.

    Args:
        filename (str): Path to the .pkl file (e.g., 'model_saver/model.pkl').
    
    Returns:
        The loaded model object, or None if an error occurs.
    """
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error loading model: {e}")
    return None


print("save_load_models.py ran successfully!")