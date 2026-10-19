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

    def generate_loss_cause_dataset(
        self,
        num_samples: int = 2000,
        is_test_split: bool = False,
        random_state: int = None
    ) -> pd.DataFrame:
        """
        Generates cross-layer network telemetry labeled with loss causes:
        NORMAL, CONGESTION, CORRUPTION, MIXED.
        Held-out test split uses different random seeds and parameter bounds to ensure generalization.
        Does not leak labels through features.
        """
        seed = random_state if random_state is not None else (999 if is_test_split else 42)
        rng = np.random.RandomState(seed)

        records = []
        labels = ["NORMAL", "CONGESTION", "CORRUPTION", "MIXED"]

        for _ in range(num_samples):
            cause = rng.choice(labels, p=[0.25, 0.25, 0.25, 0.25])
            bw = float(rng.choice([10.0, 100.0, 1000.0]))

            if not is_test_split:
                # --- TRAINING DISTRIBUTIONS ---
                if cause == "NORMAL":
                    pkt_rate = rng.uniform(50.0, 400.0)
                    q_occ = rng.uniform(0.0, 20.0)
                    base_delay = rng.uniform(15.0, 30.0)
                    delay = base_delay + (q_occ * 0.10)
                    rtt = delay * 2.0 + rng.uniform(-0.5, 0.5)
                    loss_rate = rng.uniform(0.0, 0.5)
                    active_conn = int(rng.uniform(1, 5))
                    crc_fail = rng.uniform(0.0, 0.005)
                    retrans = rng.uniform(0.0, 0.02)
                    rtt_var = rng.uniform(0.05, 1.5)
                    q_slope = rng.uniform(-0.5, 0.5)

                elif cause == "CORRUPTION":
                    # L2 CRC failures elevated, queue normal, RTT normal, no queuing slope
                    pkt_rate = rng.uniform(80.0, 450.0)
                    q_occ = rng.uniform(0.0, 25.0)
                    base_delay = rng.uniform(15.0, 32.0)
                    delay = base_delay + (q_occ * 0.10)
                    rtt = delay * 2.0 + rng.uniform(-0.5, 1.0)
                    loss_rate = rng.uniform(0.0, 1.0)
                    active_conn = int(rng.uniform(1, 6))
                    crc_fail = rng.uniform(0.08, 0.40)
                    retrans = rng.uniform(0.08, 0.42)
                    rtt_var = rng.uniform(0.1, 3.0)
                    q_slope = rng.uniform(-0.5, 0.5)

                elif cause == "CONGESTION":
                    # Queue filling, RTT inflating, queue slope positive, CRC clean
                    pkt_rate = rng.uniform(1200.0, 3000.0)
                    q_occ = rng.uniform(65.0, 100.0)
                    base_delay = rng.uniform(50.0, 85.0)
                    delay = base_delay + (q_occ * 0.45)
                    rtt = delay * 2.0 + rng.uniform(-5.0, 15.0)
                    loss_rate = rng.uniform(5.0, 25.0)
                    active_conn = int(rng.uniform(18, 50))
                    crc_fail = rng.uniform(0.0, 0.005)
                    retrans = rng.uniform(0.10, 0.45)
                    rtt_var = rng.uniform(15.0, 80.0)
                    q_slope = rng.uniform(2.0, 12.0)

                else:  # MIXED
                    # Both congestion and physical corruption present
                    pkt_rate = rng.uniform(900.0, 2600.0)
                    q_occ = rng.uniform(55.0, 95.0)
                    base_delay = rng.uniform(45.0, 80.0)
                    delay = base_delay + (q_occ * 0.40)
                    rtt = delay * 2.0 + rng.uniform(-4.0, 12.0)
                    loss_rate = rng.uniform(4.0, 20.0)
                    active_conn = int(rng.uniform(15, 45))
                    crc_fail = rng.uniform(0.06, 0.35)
                    retrans = rng.uniform(0.15, 0.55)
                    rtt_var = rng.uniform(12.0, 65.0)
                    q_slope = rng.uniform(1.5, 10.0)

            else:
                # --- HELD-OUT TEST DISTRIBUTIONS (Different parameter bounds & noise) ---
                if cause == "NORMAL":
                    pkt_rate = rng.uniform(40.0, 450.0)
                    q_occ = rng.uniform(0.0, 22.0)
                    base_delay = rng.uniform(14.0, 34.0)
                    delay = base_delay + (q_occ * 0.12)
                    rtt = delay * 2.0 + rng.uniform(-1.0, 1.0)
                    loss_rate = rng.uniform(0.0, 0.7)
                    active_conn = int(rng.uniform(1, 6))
                    crc_fail = rng.uniform(0.0, 0.008)
                    retrans = rng.uniform(0.0, 0.025)
                    rtt_var = rng.uniform(0.05, 2.0)
                    q_slope = rng.uniform(-0.8, 0.8)

                elif cause == "CORRUPTION":
                    pkt_rate = rng.uniform(70.0, 500.0)
                    q_occ = rng.uniform(0.0, 28.0)
                    base_delay = rng.uniform(16.0, 35.0)
                    delay = base_delay + (q_occ * 0.11)
                    rtt = delay * 2.0 + rng.uniform(-1.0, 1.5)
                    loss_rate = rng.uniform(0.0, 1.5)
                    active_conn = int(rng.uniform(1, 7))
                    crc_fail = rng.uniform(0.06, 0.45)
                    retrans = rng.uniform(0.07, 0.48)
                    rtt_var = rng.uniform(0.1, 4.0)
                    q_slope = rng.uniform(-0.8, 0.8)

                elif cause == "CONGESTION":
                    pkt_rate = rng.uniform(1100.0, 3200.0)
                    q_occ = rng.uniform(60.0, 100.0)
                    base_delay = rng.uniform(48.0, 90.0)
                    delay = base_delay + (q_occ * 0.48)
                    rtt = delay * 2.0 + rng.uniform(-8.0, 18.0)
                    loss_rate = rng.uniform(4.0, 28.0)
                    active_conn = int(rng.uniform(16, 55))
                    crc_fail = rng.uniform(0.0, 0.008)
                    retrans = rng.uniform(0.08, 0.50)
                    rtt_var = rng.uniform(12.0, 90.0)
                    q_slope = rng.uniform(1.8, 14.0)

                else:  # MIXED
                    pkt_rate = rng.uniform(850.0, 2800.0)
                    q_occ = rng.uniform(50.0, 98.0)
                    base_delay = rng.uniform(42.0, 85.0)
                    delay = base_delay + (q_occ * 0.42)
                    rtt = delay * 2.0 + rng.uniform(-6.0, 15.0)
                    loss_rate = rng.uniform(3.5, 24.0)
                    active_conn = int(rng.uniform(14, 50))
                    crc_fail = rng.uniform(0.05, 0.40)
                    retrans = rng.uniform(0.12, 0.60)
                    rtt_var = rng.uniform(10.0, 75.0)
                    q_slope = rng.uniform(1.2, 12.0)

            records.append({
                "bandwidth_mbps": bw,
                "packet_rate_pps": round(pkt_rate, 1),
                "queue_occupancy_pct": round(q_occ, 2),
                "rtt_ms": round(rtt, 2),
                "delay_ms": round(delay, 2),
                "loss_rate_pct": round(loss_rate, 2),
                "active_connections": active_conn,
                "crc_fail_rate": round(crc_fail, 4),
                "retransmission_rate": round(retrans, 4),
                "rtt_variance": round(rtt_var, 4),
                "queue_slope": round(q_slope, 4),
                "loss_cause_label": cause,
                "loss_cause_class": labels.index(cause)
            })

        return pd.DataFrame(records)

    def get_loss_cause_feature_matrix(
        self,
        df: pd.DataFrame,
        use_cross_layer_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts feature matrix and label vector for loss cause classification.
        When use_cross_layer_features=False, excludes L2/L4 features for ablation.
        """
        if use_cross_layer_features:
            feature_cols = [
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
            feature_cols = [
                "bandwidth_mbps",
                "packet_rate_pps",
                "queue_occupancy_pct",
                "rtt_ms",
                "loss_rate_pct",
                "active_connections"
            ]

        X = df[feature_cols].values
        y = df["loss_cause_class"].values
        return X, y

