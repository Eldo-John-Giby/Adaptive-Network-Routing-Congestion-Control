"""
Unit and Integration Tests for Cross-Layer, Loss-Cause-Aware Routing.
Tests corruption injection, classifier accuracy on held-out split, policy decisions,
hysteresis, and low-confidence fallback.
"""
import pytest
import numpy as np

from src.network.topology import NetworkTopology
from src.network.link import NetworkLink, inject_corruption
from src.network.node import RouterNode
from src.network.simulator import NetworkSimulator
from src.osi.l4_transport import TransportType
from src.ml.dataset_generator import MLDatasetGenerator
from src.ml.model import LossCauseClassifier
from src.ml.controller import LossAwareRouter, StaticDijkstraRouter, AdaptiveMLRouter


def test_corruption_injection():
    """Verifies corruption injection independent of queue stress."""
    link = NetworkLink(u="R1", v="R2", base_latency_ms=5.0)
    assert link.bit_error_rate == 0.0

    inject_corruption(link, 0.25)
    assert link.bit_error_rate == 0.25
    assert link.stress_active is False  # Queue stress must remain untouched

    # Test simulator packet traversal with corruption
    topo = NetworkTopology()
    sim = NetworkSimulator(topo)
    # Inject corruption on link S->R1
    topo.inject_link_corruption("S", "R1", bit_error_rate=0.35)

    res = sim.transmit_message(
        message="Loss Cause Telemetry Test",
        path=["S", "R1", "R2", "R4", "D"],
        transport_type=TransportType.TCP,
        max_retransmissions=4
    )
    # Router queue on R2 should not be congested
    assert topo.nodes["R2"].queue_occupancy_ratio < 0.30
    link_s_r1 = topo.get_link("S", "R1")
    assert link_s_r1.bit_error_rate == 0.35


def test_classifier_accuracy_on_held_out_split():
    """
    Measures and asserts classifier generalization on a held-out test split
    generated with different random seeds and shifted parameter bounds.
    """
    clf = LossCauseClassifier(n_estimators=100, random_state=42)
    metrics = clf.train()
    acc = metrics["accuracy"]
    importances = metrics["feature_importances"]

    # Report measured accuracy honestly
    print(f"\n[EVAL] LossCauseClassifier Held-Out Accuracy: {acc * 100:.2f}%")
    print(f"[EVAL] Feature Importances: {importances}")

    # Accuracy on unseen test split must be high (> 90%)
    assert acc >= 0.90
    assert "crc_fail_rate" in importances
    assert "queue_slope" in importances
    assert "rtt_variance" in importances
    # CRC fail rate and queue features should have high importance
    assert importances["crc_fail_rate"] > 0.05


def test_policy_decisions():
    """
    Verifies policy actions for each classified state:
    - CORRUPTION -> RETRANSMIT_SAME_PATH (no reroute)
    - CONGESTION -> REROUTE_CONGESTION (reroute to bypass)
    - MIXED -> FALLBACK_DIJKSTRA
    - NORMAL -> MAINTAIN_PATH
    """
    topo = NetworkTopology()
    router = LossAwareRouter(topology=topo, confidence_threshold=0.60, hysteresis_packets=20)

    # 1. CORRUPTION scenario: High CRC fail rate, queue is low, slope is flat
    corruption_features = {
        "bandwidth_mbps": 10.0,
        "packet_rate_pps": 180.0,
        "queue_occupancy_pct": 5.0,
        "rtt_ms": 35.0,
        "loss_rate_pct": 0.2,
        "active_connections": 2,
        "crc_fail_rate": 0.25,
        "retransmission_rate": 0.22,
        "rtt_variance": 0.8,
        "queue_slope": 0.0
    }
    path, dec = router.select_route("S", "D", features=corruption_features)
    assert dec["predicted_cause"] == "CORRUPTION"
    assert dec["action"] == "RETRANSMIT_SAME_PATH"
    assert dec["rerouted"] is False
    assert path == ["S", "R1", "R2", "R4", "D"]

    # 2. CONGESTION scenario: Clean CRC, queue is full, slope is high positive
    router.reset_state()
    congestion_features = {
        "bandwidth_mbps": 10.0,
        "packet_rate_pps": 2200.0,
        "queue_occupancy_pct": 92.0,
        "rtt_ms": 140.0,
        "loss_rate_pct": 15.0,
        "active_connections": 35,
        "crc_fail_rate": 0.0,
        "retransmission_rate": 0.28,
        "rtt_variance": 45.0,
        "queue_slope": 7.5
    }
    # Stress R2 in topology so alternate path ranking will bypass R2
    topo.stress_router_r2(True)
    path, dec = router.select_route("S", "D", features=congestion_features)
    assert dec["predicted_cause"] == "CONGESTION"
    assert dec["action"] == "REROUTE_CONGESTION"
    assert dec["rerouted"] is True
    assert "R2" not in path  # Bypassed congested node!

    # 3. MIXED scenario: High queue and high CRC failure
    router.reset_state()
    mixed_features = {
        "bandwidth_mbps": 10.0,
        "packet_rate_pps": 1600.0,
        "queue_occupancy_pct": 80.0,
        "rtt_ms": 110.0,
        "loss_rate_pct": 12.0,
        "active_connections": 25,
        "crc_fail_rate": 0.18,
        "retransmission_rate": 0.35,
        "rtt_variance": 32.0,
        "queue_slope": 4.5
    }
    path, dec = router.select_route("S", "D", features=mixed_features)
    assert dec["predicted_cause"] == "MIXED"
    assert dec["action"] == "FALLBACK_DIJKSTRA"

    # 4. NORMAL scenario: Clean network
    router.reset_state()
    normal_features = {
        "bandwidth_mbps": 10.0,
        "packet_rate_pps": 120.0,
        "queue_occupancy_pct": 8.0,
        "rtt_ms": 28.0,
        "loss_rate_pct": 0.1,
        "active_connections": 2,
        "crc_fail_rate": 0.0,
        "retransmission_rate": 0.0,
        "rtt_variance": 0.5,
        "queue_slope": 0.0
    }
    path, dec = router.select_route("S", "D", features=normal_features)
    assert dec["predicted_cause"] == "NORMAL"
    assert dec["action"] == "MAINTAIN_PATH"
    assert dec["rerouted"] is False


def test_hysteresis():
    """Verifies that after a reroute, the router holds the new path for N packets."""
    topo = NetworkTopology()
    router = LossAwareRouter(topology=topo, hysteresis_packets=10)

    # Congestion triggers a reroute
    topo.stress_router_r2(True)
    cong_feats = {
        "bandwidth_mbps": 10.0, "packet_rate_pps": 2500.0, "queue_occupancy_pct": 95.0,
        "rtt_ms": 160.0, "loss_rate_pct": 18.0, "active_connections": 38,
        "crc_fail_rate": 0.0, "retransmission_rate": 0.30, "rtt_variance": 50.0,
        "queue_slope": 8.0
    }
    path1, dec1 = router.select_route("S", "D", features=cong_feats)
    assert dec1["rerouted"] is True
    rerouted_path = list(path1)

    # Next 10 packets should be held by hysteresis, even if features change
    for i in range(10):
        p, d = router.select_route("S", "D", features=cong_feats)
        assert d["held_by_hysteresis"] is True
        assert d["action"] == "HOLD_HYSTERESIS"
        assert p == rerouted_path

    # Once 10 packets have been routed, hysteresis expires and fresh evaluation is permitted
    p_after, d_after = router.select_route("S", "D", features=cong_feats)
    assert d_after["held_by_hysteresis"] is False


def test_low_confidence_fallback():
    """Verifies that low classifier confidence triggers fallback to Dijkstra."""
    topo = NetworkTopology()
    # High confidence threshold forces fallback
    router = LossAwareRouter(topology=topo, confidence_threshold=0.99)
    router.reset_state()

    ambiguous_features = {
        "bandwidth_mbps": 10.0,
        "packet_rate_pps": 600.0,
        "queue_occupancy_pct": 40.0,
        "rtt_ms": 55.0,
        "loss_rate_pct": 2.5,
        "active_connections": 10,
        "crc_fail_rate": 0.04,
        "retransmission_rate": 0.05,
        "rtt_variance": 8.0,
        "queue_slope": 1.0
    }
    path, dec = router.select_route("S", "D", features=ambiguous_features)
    assert dec["confidence"] < 0.99
    assert dec["action"] == "FALLBACK_DIJKSTRA"
    base_dijkstra, _ = topo.dijkstra_shortest_path("S", "D", use_live_metrics=False)
    assert path == base_dijkstra
