import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

class OnlineShopperPredictor:
    """
    Complete prediction pipeline for online shopper purchase intention.

    This class handles:
    - Loading saved model and artifacts
    - Data preprocessing
    - Making predictions
    - Returning prediction probabilities
    """

    def __init__(self, models_dir='models'):
        """Initialize by loading all saved artifacts."""
        self.models_dir = Path(models_dir)
        self.load_artifacts()

    def load_artifacts(self):
        """Load model, scaler, encoders, and feature names."""
        print("Loading model artifacts...")

        # Load model
        with open(self.models_dir / 'final_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

        # Load scaler
        with open(self.models_dir / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        # Load label encoders
        with open(self.models_dir / 'label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)

        # Load feature names
        with open(self.models_dir / 'feature_names.json', 'r') as f:
            feature_info = json.load(f)
            self.numerical_features = feature_info['numerical_features']
            self.categorical_features = feature_info['categorical_features']
            self.all_features = feature_info['all_features']

        print("✓ All artifacts loaded successfully")

    def preprocess(self, X):
        """
        Preprocess input data.

        Parameters:
        -----------
        X : pd.DataFrame
            Raw input data with same structure as training data

        Returns:
        --------
        X_processed : np.ndarray
            Preprocessed and scaled features ready for prediction
        """
        X = X.copy()

        # Encode categorical features
        for col in self.categorical_features:
            if col in X.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels
                X[col] = X[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                X[col] = le.transform(X[col])

        # Ensure correct feature order
        X = X[self.all_features]

        # Scale features
        X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, X):
        """
        Make binary predictions.

        Parameters:
        -----------
        X : pd.DataFrame
            Raw input data

        Returns:
        --------
        predictions : np.ndarray
            Binary predictions (0 = No Revenue, 1 = Revenue)
        """
        X_processed = self.preprocess(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Parameters:
        -----------
        X : pd.DataFrame
            Raw input data

        Returns:
        --------
        probabilities : np.ndarray
            Probability of positive class (Revenue)
        """
        X_processed = self.preprocess(X)
        return self.model.predict_proba(X_processed)[:, 1]

    def predict_with_confidence(self, X, threshold=0.5):
        """
        Make predictions with confidence scores.

        Parameters:
        -----------
        X : pd.DataFrame
            Raw input data
        threshold : float
            Decision threshold (default=0.5)

        Returns:
        --------
        results : pd.DataFrame
            DataFrame with predictions and confidence scores
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)

        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'confidence': np.where(predictions == 1, probabilities, 1 - probabilities),
            'label': np.where(predictions == 1, 'Revenue', 'No Revenue')
        })

        return results
