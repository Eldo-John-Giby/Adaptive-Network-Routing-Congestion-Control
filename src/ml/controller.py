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
                observed_loss = (link.packet_loss_rate + link.crc_fail_rate) * 100.0
                max_loss = max(max_loss, min(100.0, observed_loss))
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

    def select_route(self, src: str = "S", dst: str = "D", features: Optional[dict] = None) -> Tuple[List[str], dict]:
        if not hasattr(self, "current_path"):
            self.current_path = None
        if not hasattr(self, "reroute_count"):
            self.reroute_count = 0
        if not hasattr(self, "needless_reroute_count"):
            self.needless_reroute_count = 0

        best_path, best_eval = self.select_best_route(src, dst)
        rerouted = False
        if self.current_path is None:
            self.current_path = list(best_path)
        elif best_path != self.current_path:
            rerouted = True
            self.reroute_count += 1
            self.current_path = list(best_path)

        return best_path, {
            "action": "ADAPTIVE_ML_REROUTE" if rerouted else "MAINTAIN_PATH",
            "chosen_path": best_path,
            "predicted_cause": best_eval.congestion_label,
            "confidence": best_eval.confidence,
            "rerouted": rerouted,
            "eval": best_eval
        }


class StaticDijkstraRouter:
    """
    Baseline Router: Always selects the static shortest path based on base propagation latency.
    Never adapts or reroutes.
    """
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.reroute_count = 0
        self.needless_reroute_count = 0
        self.current_path, _ = self.topology.dijkstra_shortest_path("S", "D", use_live_metrics=False)

    def select_route(self, src: str = "S", dst: str = "D", features: Optional[dict] = None) -> Tuple[List[str], dict]:
        path, cost = self.topology.dijkstra_shortest_path(src, dst, use_live_metrics=False)
        self.current_path = path
        return path, {
            "action": "STATIC_DIJKSTRA",
            "chosen_path": path,
            "predicted_cause": "N/A",
            "confidence": 1.0,
            "rerouted": False,
            "cost": cost
        }

StaticDijkstra = StaticDijkstraRouter


class LossAwareRouter:
    """
    Novel Cross-Layer, Loss-Cause-Aware Router.
    Classifies degradation causes using L2 (CRC) and L4 (TCP retransmission/variance) evidence,
    then executes adaptive response policies:
    - CORRUPTION: Retransmit on the same path, do not reroute.
    - CONGESTION: Reroute using path-ranking logic.
    - MIXED / LOW CONFIDENCE (< threshold): Fall back to Dijkstra base shortest path.
    - Hysteresis: Hold newly chosen paths for N packets before allowing further switches.
    """
    def __init__(
        self,
        topology: NetworkTopology,
        classifier: Optional['LossCauseClassifier'] = None,
        congestion_model: Optional[NetworkCongestionModel] = None,
        confidence_threshold: float = 0.60,
        hysteresis_packets: int = 20,
        current_path: Optional[List[str]] = None,
        use_cross_layer_features: bool = True
    ):
        from .model import LossCauseClassifier
        self.topology = topology
        self.confidence_threshold = confidence_threshold
        self.hysteresis_packets = hysteresis_packets
        self.use_cross_layer_features = use_cross_layer_features

        self.classifier = classifier or LossCauseClassifier(
            random_state=42, use_cross_layer_features=use_cross_layer_features
        )
        if not self.classifier.is_trained:
            self.classifier.train()

        self.congestion_model = congestion_model or NetworkCongestionModel()
        if not self.congestion_model.is_trained:
            self.congestion_model.train()

        # Route state
        default_dijkstra, _ = self.topology.dijkstra_shortest_path("S", "D", use_live_metrics=False)
        self.current_path: List[str] = list(current_path) if current_path else default_dijkstra
        # Initial path allows immediate reroute if stressed
        self.packets_on_current_path: int = self.hysteresis_packets
        self.reroute_count: int = 0
        self.needless_reroute_count: int = 0

    def reset_state(self):
        default_dijkstra, _ = self.topology.dijkstra_shortest_path("S", "D", use_live_metrics=False)
        self.current_path = default_dijkstra
        self.packets_on_current_path = self.hysteresis_packets
        self.reroute_count = 0
        self.needless_reroute_count = 0

    def extract_path_features(self, path: List[str]) -> dict:
        total_delay = 0.0
        max_queue_occ = 0.0
        max_loss = 0.0
        max_conn = 1
        min_bw = 1000.0
        bottleneck_node = None
        max_crc_fail_rate = 0.0
        max_retrans_rate = 0.0
        sum_rtt_variance = 0.0
        max_queue_slope = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.topology.get_link(u, v)
            if link:
                total_delay += link.calculate_link_delay()
                eff_crc = link.crc_fail_rate if link.packets_transmitted > 0 else link.bit_error_rate
                eff_retx = link.retransmission_rate if link.packets_transmitted > 0 else (link.bit_error_rate * 0.9 if link.bit_error_rate > 0 else 0.0)
                eff_var = link.rtt_variance if len(link.recent_delays) >= 2 else (2.0 if link.bit_error_rate > 0 else 0.0)
                observed_loss = (link.packet_loss_rate + eff_crc) * 100.0
                max_loss = max(max_loss, min(100.0, observed_loss))
                max_conn = max(max_conn, link.active_connections)
                min_bw = min(min_bw, link.bandwidth_mbps)
                max_crc_fail_rate = max(max_crc_fail_rate, eff_crc)
                max_retrans_rate = max(max_retrans_rate, eff_retx)
                sum_rtt_variance += eff_var

            node_v = self.topology.nodes.get(v)
            if isinstance(node_v, RouterNode):
                q_occ = node_v.queue_occupancy_ratio * 100.0
                if q_occ > max_queue_occ:
                    max_queue_occ = q_occ
                    bottleneck_node = v
                total_delay += node_v.queue_length * 2.0
                node_slope = node_v.get_queue_slope()
                if node_slope <= 0.0 and q_occ >= 65.0:
                    node_slope = 6.0
                max_queue_slope = max(max_queue_slope, node_slope)

        if max_queue_occ >= 65.0:
            if sum_rtt_variance < 10.0:
                sum_rtt_variance = max(sum_rtt_variance, 35.0)
            if max_retrans_rate < 0.10:
                max_retrans_rate = max(max_retrans_rate, 0.25)

        packet_rate = 150.0 + (max_queue_occ * 20.0)

        return {
            "bandwidth_mbps": min_bw,
            "packet_rate_pps": packet_rate,
            "queue_occupancy_pct": max_queue_occ,
            "rtt_ms": total_delay * 2.0,
            "loss_rate_pct": max_loss,
            "active_connections": max_conn,
            "crc_fail_rate": max_crc_fail_rate,
            "retransmission_rate": max_retrans_rate,
            "rtt_variance": round(sum_rtt_variance, 4),
            "queue_slope": max_queue_slope,
            "bottleneck_node": bottleneck_node,
            "accumulated_delay_ms": total_delay
        }

    def _rank_and_select_alternate_path(self, src: str = "S", dst: str = "D") -> List[str]:
        """Evaluates candidate paths using path-ranking logic and picks the best route."""
        candidate_paths = self.topology.get_candidate_paths(src, dst)
        evals = []
        for path in candidate_paths:
            feats = self.extract_path_features(path)
            cong_label, conf, pred_delay = self.congestion_model.predict_state(feats)
            penalty = 200.0 if cong_label == "HIGH" else (50.0 if cong_label == "MEDIUM" else 0.0)
            score = pred_delay + penalty
            evals.append((score, path))

        evals.sort(key=lambda x: x[0])
        return evals[0][1] if evals else self.current_path

    def select_route(
        self,
        src: str = "S",
        dst: str = "D",
        features: Optional[dict] = None
    ) -> Tuple[List[str], dict]:
        """
        Executes loss-cause classification and policy enforcement.
        Returns (chosen_path, decision_dict).
        """
        if not self.current_path:
            self.current_path, _ = self.topology.dijkstra_shortest_path(src, dst, use_live_metrics=False)

        path_features = features if features is not None else self.extract_path_features(self.current_path)
        pred_cause, confidence, probas = self.classifier.predict_cause(path_features)

        hysteresis_active = self.packets_on_current_path < self.hysteresis_packets
        action = "MAINTAIN_PATH"
        reason = ""
        chosen_path = list(self.current_path)

        if hysteresis_active:
            action = "HOLD_HYSTERESIS"
            reason = f"Hysteresis active ({self.packets_on_current_path}/{self.hysteresis_packets} pkts): holding path {self.current_path}"
            chosen_path = list(self.current_path)
        else:
            if confidence < self.confidence_threshold:
                action = "FALLBACK_DIJKSTRA"
                reason = f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}: fallback to Dijkstra"
                dijkstra_path, _ = self.topology.dijkstra_shortest_path(src, dst, use_live_metrics=False)
                chosen_path = dijkstra_path

            elif pred_cause == "CORRUPTION":
                action = "RETRANSMIT_SAME_PATH"
                reason = f"L2 CRC corruption detected (fail_rate={path_features.get('crc_fail_rate', 0):.2f}). Retransmit on current path; avoid needless reroute"
                chosen_path = list(self.current_path)

            elif pred_cause == "CONGESTION":
                action = "REROUTE_CONGESTION"
                reason = f"Congestion detected (queue={path_features.get('queue_occupancy_pct', 0):.1f}%, slope={path_features.get('queue_slope', 0):.2f}). Rerouting traffic"
                chosen_path = self._rank_and_select_alternate_path(src, dst)

            elif pred_cause == "MIXED":
                action = "FALLBACK_DIJKSTRA"
                reason = "Concurrent congestion and corruption: fallback to base Dijkstra shortest path"
                dijkstra_path, _ = self.topology.dijkstra_shortest_path(src, dst, use_live_metrics=False)
                chosen_path = dijkstra_path

            else:  # NORMAL
                action = "MAINTAIN_PATH"
                reason = "Network operating normally; optimal path maintained"
                chosen_path = list(self.current_path)

        # Update hysteresis and reroute counters
        rerouted = (chosen_path != self.current_path)
        if rerouted:
            self.current_path = list(chosen_path)
            self.packets_on_current_path = 0
            self.reroute_count += 1
            if pred_cause == "CORRUPTION":
                self.needless_reroute_count += 1
        else:
            self.packets_on_current_path += 1

        decision = {
            "action": action,
            "chosen_path": chosen_path,
            "predicted_cause": pred_cause,
            "confidence": round(confidence, 4),
            "probabilities": probas,
            "rerouted": rerouted,
            "held_by_hysteresis": hysteresis_active,
            "packets_on_current_path": self.packets_on_current_path,
            "reason": reason,
            "path_features": path_features
        }

        return chosen_path, decision

