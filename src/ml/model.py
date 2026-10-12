"""
Machine Learning Models
Trains and evaluates RandomForest models for congestion classification and delay estimation.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

from .dataset_generator import MLDatasetGenerator

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]

class NetworkCongestionModel:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False
        self.feature_names = [
            "bandwidth_mbps",
            "packet_rate_pps",
            "queue_occupancy_pct",
            "rtt_ms",
            "loss_rate_pct",
            "active_connections"
        ]
        self.metrics: Dict[str, Any] = {}

    def train(self, df: Optional[pd.DataFrame] = None, test_size: float = 0.2) -> Dict[str, Any]:
        """Trains both classifier and regressor on dataset."""
        if df is None:
            gen = MLDatasetGenerator()
            df = gen.generate_dataset(num_samples=2000)

        gen = MLDatasetGenerator()
        X, y_class, y_delay = gen.get_feature_matrix(df)

        X_train, X_test, yc_train, yc_test, yd_train, yd_test = train_test_split(
            X, y_class, y_delay, test_size=test_size, random_state=42
        )

        # 1. Train Classifier
        self.clf.fit(X_train, yc_train)
        yc_pred = self.clf.predict(X_test)
        acc = accuracy_score(yc_test, yc_pred)

        # 2. Train Regressor
        self.reg.fit(X_train, yd_train)
        yd_pred = self.reg.predict(X_test)
        mse = mean_squared_error(yd_test, yd_pred)
        r2 = r2_score(yd_test, yd_pred)

        self.is_trained = True
        self.metrics = {
            "classification_accuracy": round(acc, 4),
            "regression_mse": round(mse, 4),
            "regression_r2": round(r2, 4),
            "feature_importances": dict(zip(self.feature_names, [round(val, 4) for val in self.clf.feature_importances_]))
        }
        return self.metrics

    def predict_state(self, features: dict) -> Tuple[str, float, float]:
        """
        Input features dict:
            bandwidth_mbps, packet_rate_pps, queue_occupancy_pct, rtt_ms, loss_rate_pct, active_connections
        Returns:
            (congestion_label, confidence, predicted_delay_ms)
        """
        if not self.is_trained:
            self.train()

        feat_vector = np.array([[
            features.get("bandwidth_mbps", 10.0),
            features.get("packet_rate_pps", 100.0),
            features.get("queue_occupancy_pct", 0.0),
            features.get("rtt_ms", 30.0),
            features.get("loss_rate_pct", 0.0),
            features.get("active_connections", 1)
        ]])

        pred_class_idx = int(self.clf.predict(feat_vector)[0])
        probs = self.clf.predict_proba(feat_vector)[0]
        confidence = float(probs[pred_class_idx])
        pred_label = CLASS_NAMES[pred_class_idx]

        predicted_delay = float(self.reg.predict(feat_vector)[0])
        return pred_label, confidence, predicted_delay
