"""
Churn Prediction Module
Loads a trained PyCaret model and predicts churn probability for new customer data.
"""
import pandas as pd
import numpy as np
import pickle
import sys
from scipy import stats
from pycaret.classification import load_model, predict_model


class ChurnPredictor:
    """Loads a trained churn model and makes predictions on new data."""

    # Reverse mappings: label-encoded integers -> original string values
    # These follow alphabetical order (standard sklearn LabelEncoder)
    CONTRACT_MAP = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
    PAYMENT_MAP = {
        0: 'Bank transfer (automatic)',
        1: 'Credit card (automatic)',
        2: 'Electronic check',
        3: 'Mailed check'
    }
    PHONE_MAP = {0: 'No', 1: 'Yes'}

    def __init__(self, model_path='churn_pycaret_model',
                 distribution_path='train_prob_distribution.pkl'):
        """Load the saved PyCaret model and training probability distribution."""
        self.model = load_model(model_path)

        self.train_probs = None
        try:
            with open(distribution_path, 'rb') as f:
                self.train_probs = pickle.load(f)
        except FileNotFoundError:
            print("Note: Training distribution file not found. "
                  "Percentiles won't be available.")

    def load_data(self, filepath):
        """Load data from a CSV file into a DataFrame."""
        return pd.read_csv(filepath)

    def preprocess_raw_data(self, df):
        """
        Preprocess unmodified/raw churn data (like churn_data_week1.csv format).

        Handles the known data quality issue: TotalCharges is sometimes blank
        for brand-new customers (tenure = 0). PyCaret's pipeline handles
        the categorical encoding.
        """
        df = df.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        return df

    def _decode_labels(self, df):
        """
        Convert label-encoded integers back to original string values.
        Needed when input data uses numeric encoding (like new_churn_data.csv)
        but the PyCaret model expects original strings.
        """
        df = df.copy()
        if pd.api.types.is_numeric_dtype(df['Contract']):
            df['Contract'] = df['Contract'].map(self.CONTRACT_MAP)
        if pd.api.types.is_numeric_dtype(df['PaymentMethod']):
            df['PaymentMethod'] = df['PaymentMethod'].map(self.PAYMENT_MAP)
        if pd.api.types.is_numeric_dtype(df['PhoneService']):
            df['PhoneService'] = df['PhoneService'].map(self.PHONE_MAP)
        return df

    def _prepare(self, df):
        """
        Auto-detect input format and prepare data for the model.
        Works with both label-encoded data (new_churn_data.csv)
        and raw string data (unmodified churn data).
        """
        if pd.api.types.is_numeric_dtype(df['Contract']):
            df = self._decode_labels(df)
        else:
            df = self.preprocess_raw_data(df)

        # Drop columns the model hasn't seen
        if 'charge_per_tenure' in df.columns:
            df = df.drop('charge_per_tenure', axis=1)
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)

        return df

    def predict(self, df):
        """
        Takes a DataFrame and returns predictions with churn probabilities.
        Auto-detects whether data is label-encoded or raw strings.
        """
        prepared = self._prepare(df)
        return predict_model(self.model, data=prepared)

    def predict_proba(self, df):
        """
        Returns churn probabilities and percentile rankings for each row.

        The percentile tells you where each prediction falls in the
        distribution of training data predictions — e.g. a customer at
        the 90th percentile has a higher churn score than 90% of the
        training data.
        """
        predictions = self.predict(df)

        # PyCaret's prediction_score is the probability of the PREDICTED class.
        # If it predicted "No" (not churn), we need to invert to get churn probability.
        churn_probs = np.where(
            predictions['prediction_label'].isin(['No', 0, '0']),
            1 - predictions['prediction_score'],
            predictions['prediction_score']
        )

        result = pd.DataFrame({
            'customerID': predictions['customerID'] if 'customerID' in predictions.columns else range(len(predictions)),
            'predicted_churn': predictions['prediction_label'],
            'churn_probability': np.round(churn_probs, 4)
        })

        if self.train_probs is not None:
            result['percentile'] = [
                round(stats.percentileofscore(self.train_probs, p), 1)
                for p in churn_probs
            ]

        return result


if __name__ == "__main__":
    # Accept file path from command line or ask for it
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input(
            "Enter path to churn data CSV (default: new_churn_data.csv): "
        ).strip()
        if not filepath:
            filepath = 'new_churn_data.csv'

    predictor = ChurnPredictor()
    df = predictor.load_data(filepath)

    print(f"\nLoaded {len(df)} rows from {filepath}")
    print("\n--- Churn Predictions ---")
    results = predictor.predict_proba(df)
    print(results.to_string(index=False))
    print(f"\nTrue values: [1, 0, 0, 1, 0]")
