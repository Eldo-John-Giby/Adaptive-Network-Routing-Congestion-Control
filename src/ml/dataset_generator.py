"""
ML Dataset Generator
Generates realistic training data reflecting network telemetry across varied traffic loads.
Features: bandwidth, packet_rate, queue_length, RTT, loss_rate, active_connections.
Targets: congestion_class (LOW=0, MEDIUM=1, HIGH=2), delay_ms.
"""
import numpy as np
import pandas as pd
from typing import Tuple

class MLDatasetGenerator:
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def generate_dataset(self, num_samples: int = 1500) -> pd.DataFrame:
        """
        Creates synthetic network telemetry data covering normal, moderate, and congested states.
        """
        records = []
        for _ in range(num_samples):
            # 1. State assignment: 40% LOW, 35% MEDIUM, 25% HIGH
            state = self.rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.40, 0.35, 0.25])

            bandwidth_mbps = float(self.rng.choice([10.0, 100.0, 1000.0]))

            if state == "LOW":
                packet_rate = self.rng.uniform(50, 400)
                queue_occ = self.rng.uniform(0.0, 25.0)      # %
                loss_rate = self.rng.uniform(0.0, 0.5)       # %
                active_conn = int(self.rng.uniform(1, 5))
                base_delay = self.rng.uniform(15.0, 30.0)    # ms
                delay_ms = base_delay + (queue_occ * 0.1)
                congestion_class = 0

            elif state == "MEDIUM":
                packet_rate = self.rng.uniform(400, 1200)
                queue_occ = self.rng.uniform(25.0, 65.0)     # %
                loss_rate = self.rng.uniform(0.5, 4.0)       # %
                active_conn = int(self.rng.uniform(6, 18))
                base_delay = self.rng.uniform(30.0, 50.0)    # ms
                delay_ms = base_delay + (queue_occ * 0.25)
                congestion_class = 1

            else: # HIGH
                packet_rate = self.rng.uniform(1200, 3000)
                queue_occ = self.rng.uniform(65.0, 100.0)    # %
                loss_rate = self.rng.uniform(4.0, 25.0)      # %
                active_conn = int(self.rng.uniform(18, 50))
                base_delay = self.rng.uniform(50.0, 85.0)    # ms
                delay_ms = base_delay + (queue_occ * 0.45)
                congestion_class = 2

            records.append({
                "bandwidth_mbps": bandwidth_mbps,
                "packet_rate_pps": round(packet_rate, 1),
                "queue_occupancy_pct": round(queue_occ, 2),
                "rtt_ms": round(delay_ms * 2.0, 2),
                "delay_ms": round(delay_ms, 2),
                "loss_rate_pct": round(loss_rate, 2),
                "active_connections": active_conn,
                "congestion_class": congestion_class,
                "congestion_label": state
            })

        df = pd.DataFrame(records)
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature_cols = [
            "bandwidth_mbps",
            "packet_rate_pps",
            "queue_occupancy_pct",
            "rtt_ms",
            "loss_rate_pct",
            "active_connections"
        ]
        X = df[feature_cols].values
        y_class = df["congestion_class"].values
        y_delay = df["delay_ms"].values
        return X, y_class, y_delay
