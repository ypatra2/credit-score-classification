import logging
import pandas as pd
import numpy as np
from typing import Tuple

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

def split_data(data: pd.DataFrame, parameters: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and targets training and test sets.

    Args:
        data (pd.DataFrame): Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data.drop("credit_score", axis=1)
    y = data["credit_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Scales the data using RobustScaler and StandardScaler.

    Args:
        X_train: Training data of independent features.
        X_test: Testing data of independent features.

    Returns:
        Scaled training and testing data.
    """
    # Columns to apply RobustScaler
    robust_columns = ['total_emi_per_month', 'amount_invested_monthly', 'monthly_balance', "annual_income", "monthly_inhand_salary"]

    # Columns to apply StandardScaler (All columns except robust columns)
    standard_columns = [col for col in X_train.columns if col not in robust_columns]

    # Create the ColumnTransformer
    scaler = ColumnTransformer(
        transformers=[
            ('standard', StandardScaler(), standard_columns),
            ('robust', RobustScaler(), robust_columns)])

    # Fit on training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def train_model(X_train_scaled: np.ndarray, y_train: pd.Series, parameters: dict) -> RandomForestClassifier:
    """Trains the RandomForestClassifier model.

    Args:
        X_train_scaled: Scaled training data of independent features.
        y_train: Training data for credit score.
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        Trained RandomForestClassifier model.
    """
    
    rf_classifier = RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        random_state=parameters["random_state"],
        class_weight=parameters["class_weight"]
    )
    rf_classifier.fit(X_train_scaled, y_train)
    return rf_classifier

def evaluate_model(
    rf_classifier: RandomForestClassifier, X_test_scaled: pd.DataFrame, y_test: pd.Series
) -> None:
    """Predicts the credit scores using the trained RandomForestClassifier and logs the confusion matrices and classification report for training and testing data.

    Args:
        rf_classifier: Trained RandomForestClassifier model.
        X_test_scaled: Scaled testing data of independent features.
        y_test: Testing data for credit score.
    """
    # Predict the target variable
    y_pred = rf_classifier.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy}")

    # Print classification report
    logger.info("Classification Report:\n")
    logger.info(classification_report(y_test, y_pred))

    # Print confusion matrix
    logger.info("Confusion Matrix:\n")
    logger.info(confusion_matrix(y_test, y_pred))
