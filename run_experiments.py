"""
Comprehensive Experimentation Harness
Evaluates Loss-Cause-Aware Routing against baselines:
1. Static Dijkstra
2. Adaptive ML Router (baseline)
3. Loss-Aware Router (Ours)
4. Loss-Aware Router (Ablation: without L2/L4 features)

Across 5 distinct network scenarios:
- Clean network
- Corruption-only on one link
- Congestion-only on R2
- Congestion on R1
- Both at once (congested R2 + corrupted R1->R3)

Outputs:
- CSV: results/experiment_results.csv
- Plotly interactive figures: results/*.html
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure root directory in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.network.topology import NetworkTopology
from src.network.simulator import NetworkSimulator
from src.osi.l4_transport import TransportType
from src.ml.model import NetworkCongestionModel, LossCauseClassifier
from src.ml.controller import StaticDijkstraRouter, AdaptiveMLRouter, LossAwareRouter


def setup_scenario(scenario_name: str) -> NetworkTopology:
    """Creates and configures a network topology for the given scenario."""
    topo = NetworkTopology()

    if scenario_name == "clean_network":
        pass  # Baseline clean network

    elif scenario_name == "corruption_only":
        # Inject bit corruption on R1<->R2 link (BER = 0.20), no queue congestion
        topo.inject_link_corruption("R1", "R2", bit_error_rate=0.20)

    elif scenario_name == "congestion_only_r2":
        # Stress router R2 (queue filled, delays multiplied)
        topo.stress_router_r2(True)

    elif scenario_name == "congestion_r1":
        # Stress router R1 (queue filled, connected links stressed)
        topo.stress_router_r1(True)

    elif scenario_name == "both_at_once":
        # R2 congested + R1<->R3 suffers bit corruption (BER = 0.20)
        topo.stress_router_r2(True)
        topo.inject_link_corruption("R1", "R3", bit_error_rate=0.20)

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    return topo


def run_single_experiment(
    router_type: str,
    scenario_name: str,
    num_packets: int = 100,
    seed: int = 42
) -> dict:
    """Runs a single simulation experiment with fixed seed and collects metrics."""
    random.seed(seed)
    np.random.seed(seed)

    topo = setup_scenario(scenario_name)
    sim = NetworkSimulator(topo)

    # Initialize requested router
    if router_type == "StaticDijkstra":
        router = StaticDijkstraRouter(topo)
    elif router_type == "AdaptiveMLRouter":
        router = AdaptiveMLRouter(topo)
    elif router_type == "LossAwareRouter":
        router = LossAwareRouter(topo, use_cross_layer_features=True, hysteresis_packets=20)
    elif router_type == "LossAwareRouter_Ablation":
        router = LossAwareRouter(topo, use_cross_layer_features=False, hysteresis_packets=20)
    else:
        raise ValueError(f"Unknown router: {router_type}")

    latencies = []
    delivered_flags = []
    retransmission_counts = []
    reroute_events = 0
    consecutive_success = 0
    time_to_recover = num_packets  # packets until stability

    initial_path = None

    for pkt_idx in range(num_packets):
        # Select route
        path, decision = router.select_route("S", "D")
        if pkt_idx == 0:
            initial_path = list(path)

        rerouted = decision.get("rerouted", False)
        if rerouted:
            reroute_events += 1

        # Transmit message across chosen path
        res = sim.transmit_message(
            message=f"EXP_DATA_PKT_{pkt_idx}",
            path=path,
            transport_type=TransportType.TCP,
            max_retransmissions=3
        )

        latencies.append(res.total_latency_ms)
        delivered_flags.append(res.delivered)
        retransmission_counts.append(res.retransmissions)

        # Track stability/recovery
        if res.delivered:
            consecutive_success += 1
            if consecutive_success >= 5 and time_to_recover == num_packets:
                time_to_recover = pkt_idx + 1
        else:
            consecutive_success = 0

    # Calculate needless reroutes
    # In clean network: every reroute away is needless
    # In corruption-only: every reroute away from best path (Route 1) is needless
    # In congestion scenarios: the initial reroute away from congested router is valid;
    # subsequent reroutes (flapping) are needless.
    if scenario_name in ["clean_network", "corruption_only"]:
        needless_reroutes = reroute_events
    elif scenario_name in ["congestion_only_r2", "congestion_r1"]:
        needless_reroutes = max(0, reroute_events - 1)
    else:  # both_at_once
        needless_reroutes = max(0, reroute_events - 1)

    mean_lat = round(float(np.mean(latencies)), 2)
    p95_lat = round(float(np.percentile(latencies, 95)), 2)
    deliv_ratio = round(sum(delivered_flags) / len(delivered_flags), 4)
    total_retrans = int(sum(retransmission_counts))

    return {
        "router": router_type,
        "scenario": scenario_name,
        "packets_sent": num_packets,
        "needless_reroutes": needless_reroutes,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "delivered_ratio": deliv_ratio,
        "total_retransmissions": total_retrans,
        "time_to_recover_pkts": time_to_recover
    }


def run_all_experiments() -> pd.DataFrame:
    scenarios = [
        "clean_network",
        "corruption_only",
        "congestion_only_r2",
        "congestion_r1",
        "both_at_once"
    ]
    routers = [
        "StaticDijkstra",
        "AdaptiveMLRouter",
        "LossAwareRouter",
        "LossAwareRouter_Ablation"
    ]

    print("=" * 85)
    print("RUNNING LOSS-CAUSE-AWARE CROSS-LAYER ROUTING EXPERIMENTS")
    print("Scenarios: 5 | Routers: 4 | Packets/Run: 100 | Seed: 42")
    print("=" * 85)

    results = []
    for sc in scenarios:
        print(f"\n[+] Scenario: {sc.upper()}")
        for r in routers:
            t0 = time.time()
            res = run_single_experiment(r, sc, num_packets=100, seed=42)
            elapsed = time.time() - t0
            results.append(res)
            print(f"    - {r:26s} | Latency: {res['mean_latency_ms']:5.1f}ms (p95: {res['p95_latency_ms']:5.1f}ms) | Retrans: {res['total_retransmissions']:2d} | Deliv: {res['delivered_ratio']*100:5.1f}% | Needless Reroutes: {res['needless_reroutes']}")

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Raw experiment results saved to: {csv_path}")

    # Generate Figures
    generate_figures(df)
    return df


def generate_figures(df: pd.DataFrame):
    """Generates Plotly interactive figures for research visual analysis."""
    os.makedirs("results", exist_ok=True)
    colors = {
        "StaticDijkstra": "#64748B",
        "AdaptiveMLRouter": "#EF4444",
        "LossAwareRouter": "#10B981",
        "LossAwareRouter_Ablation": "#F59E0B"
    }

    # 1. Needless Reroutes Comparison
    fig_reroutes = px.bar(
        df,
        x="scenario",
        y="needless_reroutes",
        color="router",
        barmode="group",
        color_discrete_map=colors,
        title="Needless Reroute Count by Scenario and Router",
        labels={"needless_reroutes": "Needless Reroutes", "scenario": "Scenario", "router": "Routing Strategy"}
    )
    fig_reroutes.update_layout(template="plotly_dark", font=dict(family="JetBrains Mono, sans-serif"))
    fig_reroutes.write_html(os.path.join("results", "needless_reroutes.html"))

    # 2. Mean and P95 Latency Comparison
    fig_lat = px.bar(
        df,
        x="scenario",
        y="mean_latency_ms",
        color="router",
        barmode="group",
        color_discrete_map=colors,
        title="Mean End-to-End Latency (ms) by Scenario and Router",
        labels={"mean_latency_ms": "Mean Latency (ms)", "scenario": "Scenario"}
    )
    fig_lat.update_layout(template="plotly_dark", font=dict(family="JetBrains Mono, sans-serif"))
    fig_lat.write_html(os.path.join("results", "latency_comparison.html"))

    # 3. Delivery Ratio Comparison
    fig_deliv = px.bar(
        df,
        x="scenario",
        y="delivered_ratio",
        color="router",
        barmode="group",
        color_discrete_map=colors,
        title="Delivered Packet Ratio by Scenario and Router",
        labels={"delivered_ratio": "Delivery Ratio", "scenario": "Scenario"}
    )
    fig_deliv.update_layout(template="plotly_dark", font=dict(family="JetBrains Mono, sans-serif"))
    fig_deliv.write_html(os.path.join("results", "delivered_ratio.html"))

    # 4. Summary Multi-Panel Dashboard
    fig_summary = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Needless Reroutes (Lower is Better)",
            "Mean Latency in ms (Lower is Better)",
            "Packet Delivery Ratio (Higher is Better)",
            "Total Retransmissions (Lower is Better)"
        )
    )
    for r_name in df["router"].unique():
        sub_df = df[df["router"] == r_name]
        fig_summary.add_trace(go.Bar(name=r_name, x=sub_df["scenario"], y=sub_df["needless_reroutes"], marker_color=colors[r_name]), row=1, col=1)
        fig_summary.add_trace(go.Bar(name=r_name, x=sub_df["scenario"], y=sub_df["mean_latency_ms"], marker_color=colors[r_name], showlegend=False), row=1, col=2)
        fig_summary.add_trace(go.Bar(name=r_name, x=sub_df["scenario"], y=sub_df["delivered_ratio"], marker_color=colors[r_name], showlegend=False), row=2, col=1)
        fig_summary.add_trace(go.Bar(name=r_name, x=sub_df["scenario"], y=sub_df["total_retransmissions"], marker_color=colors[r_name], showlegend=False), row=2, col=2)

    fig_summary.update_layout(
        title_text="Cross-Layer Loss-Cause-Aware Router: Multi-Metric Benchmark Suite",
        template="plotly_dark",
        height=800,
        font=dict(family="JetBrains Mono, sans-serif")
    )
    fig_summary.write_html(os.path.join("results", "summary_barchart.html"))
    print("[OK] Plotly visual figures saved to: results/*.html")


if __name__ == "__main__":
    df_results = run_all_experiments()
    print("\n" + "=" * 85)
    print("FINAL EXPERIMENT RESULTS TABLE:")
    print("=" * 85)
    print(df_results.to_string(index=False))
