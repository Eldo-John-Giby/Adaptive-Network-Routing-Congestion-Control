"""
Unit tests for Network Topology, Queues, ML Models and Dynamic Routing
"""
import pytest
from src.network.topology import NetworkTopology
from src.network.node import RouterNode
from src.ml.dataset_generator import MLDatasetGenerator
from src.ml.model import NetworkCongestionModel
from src.ml.controller import AdaptiveMLRouter

def test_topology_paths_and_costs():
    topo = NetworkTopology()
    paths = topo.get_candidate_paths("S", "D")
    # All 3 paths from Figure 5 should be present:
    # 1. S -> R1 -> R2 -> R4 -> D
    # 2. S -> R1 -> R3 -> R4 -> D
    # 3. S -> R3 -> R4 -> D
    path_strs = ["->".join(p) for p in paths]
    assert "S->R1->R2->R4->D" in path_strs
    assert "S->R1->R3->R4->D" in path_strs
    assert "S->R3->R4->D" in path_strs

    # Base cost for S->R1->R2->R4->D: 8 + 5 + 4 + 7 = 24
    c1 = topo.calculate_path_cost(["S", "R1", "R2", "R4", "D"])
    assert c1 == 24.0

def test_router_queue_drop_tail():
    router = RouterNode("R1", "10.0.1.1", "00:00:00:00:00:02", max_queue_size=5)
    for i in range(5):
        assert router.enqueue_packet(f"pkt_{i}") is True
    # 6th packet should drop (overflow)
    assert router.enqueue_packet("overflow_pkt") is False
    assert router.packets_dropped == 1

def test_ml_dataset_generator():
    gen = MLDatasetGenerator()
    df = gen.generate_dataset(num_samples=200)
    assert len(df) == 200
    assert "congestion_class" in df.columns
    assert "delay_ms" in df.columns
    assert set(df["congestion_class"].unique()).issubset({0, 1, 2})

def test_ml_model_training_and_accuracy():
    model = NetworkCongestionModel(n_estimators=30)
    metrics = model.train()
    assert metrics["classification_accuracy"] >= 0.85
    assert metrics["regression_r2"] >= 0.70

    # Test prediction on congested sample
    congested_features = {
        "bandwidth_mbps": 10.0,
        "packet_rate_pps": 2500.0,
        "queue_occupancy_pct": 95.0,
        "rtt_ms": 150.0,
        "loss_rate_pct": 18.0,
        "active_connections": 35
    }
    label, conf, delay = model.predict_state(congested_features)
    assert label == "HIGH"
    assert delay > 50.0

def test_adaptive_rerouting_under_r2_stress():
    topo = NetworkTopology()
    router = AdaptiveMLRouter(topology=topo)

    # Initially, with no stress, Route 1 (via R2) is valid
    evals_normal = router.evaluate_all_candidate_paths("S", "D")
    r1_eval = next(e for e in evals_normal if e.path == ["S", "R1", "R2", "R4", "D"])
    assert r1_eval.congestion_label in ["LOW", "MEDIUM"]

    # Now congest Router R2
    topo.stress_router_r2(stress_active=True)
    evals_stressed = router.evaluate_all_candidate_paths("S", "D")
    r1_eval_stressed = next(e for e in evals_stressed if e.path == ["S", "R1", "R2", "R4", "D"])

    # ML should identify Route 1 as HIGH congestion and recommend avoiding it
    assert r1_eval_stressed.congestion_label == "HIGH"
    assert "Avoid" in r1_eval_stressed.recommendation

    # Selected best route must bypass R2!
    best_path, best_eval = router.select_best_route("S", "D")
    assert "R2" not in best_path
    assert best_eval.recommendation == "Preferred"
