"""
Adaptive ML Controller
Implements the closed-loop decision engine that extracts live network metrics,
runs ML model inference across candidate paths, and adapts routing paths dynamically.
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from ..network.topology import NetworkTopology
from ..network.node import RouterNode
from .model import NetworkCongestionModel

@dataclass
class RouteEvaluation:
    path: List[str]
    path_name: str
    congestion_label: str       # "LOW", "MEDIUM", "HIGH"
    confidence: float
    predicted_delay_ms: float
    actual_base_delay_ms: float
    recommendation: str         # "Preferred", "Backup", "Avoid (High Congestion)"
    bottleneck_node: Optional[str] = None

class AdaptiveMLRouter:
    def __init__(self, topology: NetworkTopology, model: Optional[NetworkCongestionModel] = None):
        self.topology = topology
        self.model = model or NetworkCongestionModel()
        if not self.model.is_trained:
            self.model.train()

    def extract_path_features(self, path: List[str]) -> dict:
        """
        Aggregates real-time features along candidate path for ML model ingestion.
        """
        total_delay = 0.0
        max_queue_occ = 0.0
        max_loss = 0.0
        max_conn = 1
        min_bw = 1000.0
        bottleneck_node = None

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.topology.get_link(u, v)
            if link:
                total_delay += link.calculate_link_delay()
                max_loss = max(max_loss, link.packet_loss_rate * 100.0)
                max_conn = max(max_conn, link.active_connections)
                min_bw = min(min_bw, link.bandwidth_mbps)

            node_v = self.topology.nodes.get(v)
            if isinstance(node_v, RouterNode):
                q_occ = node_v.queue_occupancy_ratio * 100.0
                if q_occ > max_queue_occ:
                    max_queue_occ = q_occ
                    bottleneck_node = v
                total_delay += node_v.queue_length * 2.0

        packet_rate = 150.0 + (max_queue_occ * 20.0)

        return {
            "bandwidth_mbps": min_bw,
            "packet_rate_pps": packet_rate,
            "queue_occupancy_pct": max_queue_occ,
            "rtt_ms": total_delay * 2.0,
            "loss_rate_pct": max_loss,
            "active_connections": max_conn,
            "bottleneck_node": bottleneck_node,
            "accumulated_delay_ms": total_delay
        }

    def evaluate_all_candidate_paths(self, src: str = "S", dst: str = "D") -> List[RouteEvaluation]:
        candidate_paths = self.topology.get_candidate_paths(src, dst)
        evaluations: List[RouteEvaluation] = []

        path_labels = {
            "S->R1->R2->R4->D": "Route 1 (via R2)",
            "S->R1->R3->R4->D": "Route 2 (via R3)",
            "S->R3->R4->D": "Route 3 (direct R3)"
        }

        for path in candidate_paths:
            path_key = "->".join(path)
            path_name = path_labels.get(path_key, path_key)
            features = self.extract_path_features(path)
            congestion_label, confidence, pred_delay = self.model.predict_state(features)
            base_delay = self.topology.calculate_path_cost(path, use_live_metrics=False)

            evaluations.append(RouteEvaluation(
                path=path,
                path_name=path_name,
                congestion_label=congestion_label,
                confidence=round(confidence, 3),
                predicted_delay_ms=round(pred_delay, 1),
                actual_base_delay_ms=round(base_delay, 1),
                recommendation="",
                bottleneck_node=features["bottleneck_node"]
            ))

        # Sort paths by predicted delay and congestion state
        # Penalty ranking: HIGH congestion gets pushed down
        def route_rank(ev: RouteEvaluation):
            penalty = 200.0 if ev.congestion_label == "HIGH" else (50.0 if ev.congestion_label == "MEDIUM" else 0.0)
            return ev.predicted_delay_ms + penalty

        evaluations.sort(key=route_rank)

        # Assign recommendations
        if evaluations:
            evaluations[0].recommendation = "Preferred"
            if len(evaluations) > 1:
                evaluations[1].recommendation = "Backup"
            for ev in evaluations[2:]:
                ev.recommendation = "Avoid (High Congestion)" if ev.congestion_label == "HIGH" else "Alternate"

            # Check if any path marked HIGH still had preferred/backup status
            for ev in evaluations:
                if ev.congestion_label == "HIGH":
                    ev.recommendation = "Avoid if congestion is high"

        return evaluations

    def select_best_route(self, src: str = "S", dst: str = "D") -> Tuple[List[str], RouteEvaluation]:
        evals = self.evaluate_all_candidate_paths(src, dst)
        # Select best non-congested route, or lowest predicted delay
        best_eval = evals[0]
        return best_eval.path, best_eval
