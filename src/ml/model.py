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

LOSS_CAUSE_CLASSES = ["NORMAL", "CONGESTION", "CORRUPTION", "MIXED"]

class LossCauseClassifier:
    """
    Random Forest Classifier (100 estimators) classifying the root cause of packet degradation:
    - NORMAL: clean path, baseline operational telemetry
    - CONGESTION: queue build-up, delay inflation, zero CRC errors
    - CORRUPTION: L2 CRC check failures, normal queues and delays
    - MIXED: concurrent congestion and channel corruption
    """
    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        use_cross_layer_features: bool = True
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_cross_layer_features = use_cross_layer_features
        if use_cross_layer_features:
            self.feature_names = [
                "bandwidth_mbps",
                "packet_rate_pps",
                "queue_occupancy_pct",
                "rtt_ms",
                "loss_rate_pct",
                "active_connections",
                "crc_fail_rate",
                "retransmission_rate",
                "rtt_variance",
                "queue_slope"
            ]
        else:
            # Ablation: without L2 (crc_fail_rate) and L4 (retransmission_rate, rtt_variance)
            self.feature_names = [
                "bandwidth_mbps",
                "packet_rate_pps",
                "queue_occupancy_pct",
                "rtt_ms",
                "loss_rate_pct",
                "active_connections"
            ]

        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}

    def train(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        gen = MLDatasetGenerator(random_state=self.random_state)
        if train_df is None:
            train_df = gen.generate_loss_cause_dataset(num_samples=2500, is_test_split=False)
        if test_df is None:
            test_df = gen.generate_loss_cause_dataset(num_samples=600, is_test_split=True)

        X_train, y_train = gen.get_loss_cause_feature_matrix(train_df, self.use_cross_layer_features)
        X_test, y_test = gen.get_loss_cause_feature_matrix(test_df, self.use_cross_layer_features)

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=LOSS_CAUSE_CLASSES, output_dict=True, zero_division=0)

        self.is_trained = True
        self.metrics = {
            "accuracy": round(float(acc), 4),
            "classification_report": report,
            "feature_importances": self.feature_importances
        }
        return self.metrics

    @property
    def feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        return dict(zip(
            self.feature_names,
            [round(float(v), 4) for v in self.clf.feature_importances_]
        ))

    def _build_feature_vector(self, features: dict) -> np.ndarray:
        defaults = {
            "bandwidth_mbps": 10.0,
            "packet_rate_pps": 150.0,
            "queue_occupancy_pct": 0.0,
            "rtt_ms": 30.0,
            "loss_rate_pct": 0.0,
            "active_connections": 1,
            "crc_fail_rate": 0.0,
            "retransmission_rate": 0.0,
            "rtt_variance": 0.0,
            "queue_slope": 0.0
        }
        row = [float(features.get(k, defaults.get(k, 0.0))) for k in self.feature_names]
        return np.array([row])

    def predict(self, X) -> np.ndarray:
        if not self.is_trained:
            self.train()
        if isinstance(X, dict):
            X = self._build_feature_vector(X)
        return self.clf.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_trained:
            self.train()
        if isinstance(X, dict):
            X = self._build_feature_vector(X)
        return self.clf.predict_proba(X)

    def predict_cause(self, features: dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Input features dict:
            bandwidth_mbps, packet_rate_pps, queue_occupancy_pct, rtt_ms, loss_rate_pct,
            active_connections, crc_fail_rate, retransmission_rate, rtt_variance, queue_slope
        Returns:
            (predicted_cause_label, confidence, proba_dict)
        """
        if not self.is_trained:
            self.train()

        feat_vector = self._build_feature_vector(features)
        probs = self.clf.predict_proba(feat_vector)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = LOSS_CAUSE_CLASSES[pred_idx]

        proba_dict = {
            LOSS_CAUSE_CLASSES[i]: round(float(probs[i]), 4)
            for i in range(len(LOSS_CAUSE_CLASSES))
        }
        return pred_label, confidence, proba_dict

