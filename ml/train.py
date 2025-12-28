"""
Machine Learning Training Pipeline
Handles model training, validation, and evaluation
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """Machine Learning training pipeline for model development and evaluation."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration dictionary for pipeline parameters
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler = None
        self.metrics = {}
        logger.info("Training pipeline initialized")

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'scaling': True,
        }

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data from file.
        
        Args:
            data_path: Path to the data file (CSV, JSON, or Parquet)
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif path.suffix == '.json':
            df = pd.read_json(data_path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def preprocess_data(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, pd.Series]:
        """
        Preprocess features and labels.
        
        Args:
            X: Feature matrix
            y: Target variable
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Tuple of preprocessed features and labels
        """
        logger.info("Preprocessing data")
        
        # Handle missing values
        X_processed = X.fillna(X.mean(numeric_only=True))
        
        # Scale features
        if self.config['scaling']:
            if fit_scaler:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X_processed)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted. Fit training data first.")
                X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        logger.info("Data preprocessing completed")
        return X_scaled, y

    def split_data(
        self,
        X: np.ndarray,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y if hasattr(y, 'value_counts') else None
        )
        
        logger.info(
            f"Train set: {X_train.shape[0]} samples, "
            f"Test set: {X_test.shape[0]} samples"
        )
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Train the machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Starting model training")
        
        self.model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            random_state=self.config['random_state'],
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating model performance")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        logger.info(f"Model evaluation metrics: {self.metrics}")
        return self.metrics

    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def save_model(self, model_path: str) -> None:
        """
        Save trained model and scaler to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {model_path}")

    def save_metrics(self, metrics_path: str) -> None:
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            metrics_path: Path to save metrics
        """
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_json = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in self.metrics.items()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_path}")


def main():
    """Main training pipeline execution."""
    logger.info("Starting ML training pipeline")
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline()
    
    # Example usage (requires actual data)
    # Uncomment and modify for actual usage:
    # 
    # data = pipeline.load_data('data/train.csv')
    # X = data.drop('target', axis=1)
    # y = data['target']
    # 
    # X_processed, y = pipeline.preprocess_data(X, y, fit_scaler=True)
    # X_train, X_test, y_train, y_test = pipeline.split_data(X_processed, y)
    # 
    # pipeline.train_model(X_train, y_train)
    # metrics = pipeline.evaluate_model(X_test, y_test)
    # 
    # feature_importance = pipeline.get_feature_importance(X.columns.tolist())
    # print(feature_importance)
    # 
    # pipeline.save_model('models/model.pkl')
    # pipeline.save_metrics('metrics/results.json')
    
    logger.info("ML training pipeline completed")


if __name__ == '__main__':
    main()
