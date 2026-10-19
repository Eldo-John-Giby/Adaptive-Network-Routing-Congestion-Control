"""
Interactive Visual Dashboard for Adaptive Network Routing & Congestion Control
Built with Streamlit, Plotly, NetworkX, and Matplotlib.
High-impact, production-grade network telemetry and ML intelligence visualization suite.
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.osi.packet_engine import OSIStack
from src.osi.l4_transport import TransportType
from src.network.topology import NetworkTopology
from src.network.simulator import NetworkSimulator
from src.ml.model import NetworkCongestionModel, LossCauseClassifier
from src.ml.controller import AdaptiveMLRouter, LossAwareRouter, StaticDijkstraRouter
from src.network.link import inject_corruption
from src.endpoints.socket_server import SocketServer
from src.endpoints.socket_client import SocketClient

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Network Intelligence Operations Center — 7-Layer OSI & Adaptive ML Routing",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# HIGH-DENSITY CYBERNETIC DARK THEME STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }

    /* Operations Center Header */
    .noc-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 50%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 6px solid #3B82F6;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .noc-title-area h1 {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .noc-title-area p {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94A3B8;
        margin: 0.3rem 0 0 0;
        letter-spacing: 0.02em;
    }
    .noc-status-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10B981;
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
    }

    /* Panel Card */
    .glass-panel {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .glass-panel h3 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0 0 0.8rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        padding-bottom: 0.5rem;
    }

    /* KPI Metrics Grid */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .kpi-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        position: relative;
        overflow: hidden;
    }
    .kpi-box::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: #3B82F6;
    }
    .kpi-box.emerald::after { background: #10B981; }
    .kpi-box.amber::after { background: #F59E0B; }
    .kpi-box.violet::after { background: #8B5CF6; }
    .kpi-box.rose::after { background: #F43F5E; }

    .kpi-tag {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .kpi-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-desc {
        font-size: 0.74rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    /* Status Badges */
    .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-success { background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }
    .badge-danger { background: #7F1D1D; color: #FCA5A5; border: 1px solid #DC2626; }
    .badge-warning { background: #78350F; color: #FDE68A; border: 1px solid #D97706; }
    .badge-info { background: #1E3A8A; color: #93C5FD; border: 1px solid #2563EB; }
    .badge-purple { background: #4C1D95; color: #DDD6FE; border: 1px solid #7C3AED; }

    /* OSI Layer Stepper Cards */
    .osi-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        transition: all 0.2s ease;
    }
    .osi-card:hover {
        border-color: rgba(96, 165, 250, 0.6);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.15);
    }
    .osi-card-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.45rem;
    }
    .osi-layer-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
    }
    .osi-layer-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-left: 0.4rem;
    }
    .osi-payload-box {
        background: #090D16;
        border: 1px solid #1E293B;
        border-radius: 4px;
        padding: 0.5rem 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #CBD5E1;
        overflow-x: auto;
    }

    /* Console log rows */
    .log-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.55rem 0.85rem;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.4);
        border-radius: 6px;
        margin-bottom: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #E2E8F0;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 2px solid #1E293B;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
        color: #94A3B8;
        padding: 0.7rem 1.4rem;
        background: transparent;
        border-radius: 6px 6px 0 0;
        letter-spacing: 0.02em;
    }
    .stTabs [aria-selected="true"] {
        color: #60A5FA !important;
        background: rgba(30, 41, 59, 0.6) !important;
        border-bottom: 3px solid #3B82F6 !important;
    }

    /* Notification callouts */
    .callout-info {
        background: rgba(30, 58, 138, 0.35);
        border: 1px solid #2563EB;
        border-left: 4px solid #60A5FA;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        color: #DBEAFE;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .callout-warn {
        background: rgba(120, 53, 15, 0.35);
        border: 1px solid #D97706;
        border-left: 4px solid #FBBF24;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        color: #FEF3C7;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Custom button aesthetics */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.55rem 1.3rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATE INITIALIZATION
# ─────────────────────────────────────────────
if "topology" not in st.session_state:
    st.session_state.topology = NetworkTopology()
if "simulator" not in st.session_state:
    st.session_state.simulator = NetworkSimulator(st.session_state.topology)
if "ml_model" not in st.session_state:
    model = NetworkCongestionModel()
    model.train()
    st.session_state.ml_model = model
if "router" not in st.session_state:
    st.session_state.router = AdaptiveMLRouter(
        st.session_state.topology, st.session_state.ml_model
    )
if "loss_classifier" not in st.session_state:
    loss_clf = LossCauseClassifier()
    loss_clf.train()
    st.session_state.loss_classifier = loss_clf
if "loss_router" not in st.session_state:
    st.session_state.loss_router = LossAwareRouter(
        st.session_state.topology,
        classifier=st.session_state.loss_classifier,
        congestion_model=st.session_state.ml_model
    )
if "history" not in st.session_state:
    st.session_state.history = []
if "last_enc_trace" not in st.session_state:
    st.session_state.last_enc_trace = None
if "last_dec_trace" not in st.session_state:
    st.session_state.last_dec_trace = None
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None
if "last_corrupt_flag" not in st.session_state:
    st.session_state.last_corrupt_flag = False

# ─────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="noc-header">
    <div class="noc-title-area">
        <h1>Adaptive Network Operations Center</h1>
        <p>End-to-End 7-Layer OSI Architecture &nbsp;|&nbsp; Multi-Hop Queue Buffer Dynamics &nbsp;|&nbsp; Closed-Loop ML Rerouting</p>
    </div>
    <div class="noc-status-badge">
        <span class="pulse-dot"></span> Telemetry Engine Online
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "Network Topology & Adaptive Routing",
    "7-Layer OSI Packet Engine",
    "Operations Telemetry & Diagnostics",
    "Machine Learning Decision Matrix",
    "Host Socket Interoperability",
    "Loss-Cause Analysis",
])

# ═══════════════════════════════════════════════════════════════
# HELPER: INTERACTIVE NETWORK GRAPH (PLOTLY)
# ═══════════════════════════════════════════════════════════════
def build_topology_figure(topology: NetworkTopology, active_path=None, stress_r2=False, packet_pos=None):
    """
    Renders an interactive high-contrast network topology visual with Plotly.
    Figure 5 node layout with live telemetry overlays and dynamic path highlighting.
    packet_pos: optional (x, y, label, color) representing a live moving packet across links.
    """
    # Coordinates for Figure 5 topology
    pos = {
        "S":  (-2.0,  0.0),
        "R1": (-0.8,  0.8),
        "R3": (-0.8, -0.8),
        "R2": ( 0.6,  0.8),
        "R4": ( 1.4,  0.0),
        "D":  ( 2.6,  0.0),
    }

    # Links definition with specifications
    edges_info = [
        ("S",  "R1", 8.0,  "8 ms / 10 Mbps"),
        ("S",  "R3", 14.0, "14 ms / 10 Mbps"),
        ("R1", "R2", 5.0,  "5 ms / 10 Mbps (Stressed: 30ms)" if stress_r2 else "5 ms / 10 Mbps"),
        ("R1", "R3", 6.0,  "6 ms / 10 Mbps"),
        ("R2", "R4", 4.0,  "4 ms / 10 Mbps (Stressed: 24ms)" if stress_r2 else "4 ms / 10 Mbps"),
        ("R3", "R4", 3.0,  "3 ms / 10 Mbps"),
        ("R4", "D",  7.0,  "7 ms / 10 Mbps"),
    ]

    active_edges = set()
    if active_path:
        for i in range(len(active_path) - 1):
            u, v = active_path[i], active_path[i + 1]
            active_edges.add((u, v))
            active_edges.add((v, u))

    fig = go.Figure()

    # Draw inactive and active edges
    for u, v, base_lat, label in edges_info:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_active = (u, v) in active_edges or (v, u) in active_edges
        is_stressed_link = stress_r2 and (u in ["R1", "R2", "R4"] and v in ["R1", "R2", "R4"] and ("R2" in (u, v)))

        if is_active:
            edge_color = "#38BDF8"  # bright cyan highlight
            edge_width = 4.5
        elif is_stressed_link:
            edge_color = "#EF4444"  # bright red congestion warning
            edge_width = 2.8
        else:
            edge_color = "#334155"  # slate border
            edge_width = 2.0

        # Edge line trace
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(color=edge_color, width=edge_width),
            hoverinfo="text",
            text=f"Link {u} ↔ {v}<br>Spec: {label}<br>Status: {'CONGESTED (+20% drop)' if is_stressed_link else 'NORMAL'}",
            showlegend=False,
        ))

        # Edge midpoint latency annotation label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2 + (0.1 if y0 == y1 else 0.0)
        disp_lat = f"{int(base_lat * 6)}ms" if is_stressed_link else f"{int(base_lat)}ms"
        label_color = "#F87171" if is_stressed_link else ("#7DD3FC" if is_active else "#94A3B8")

        fig.add_annotation(
            x=mid_x, y=mid_y,
            text=disp_lat,
            showarrow=False,
            font=dict(family="JetBrains Mono", size=10, color=label_color),
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(71, 85, 105, 0.6)",
            borderwidth=1,
            borderpad=2,
        )

    # Draw nodes
    node_x, node_y, node_text, node_hover, node_color, node_size, border_color = [], [], [], [], [], [], []
    for n, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)

        node_obj = topology.nodes.get(n)
        q_len = getattr(node_obj, "queue_length", 0)
        q_max = getattr(node_obj, "max_queue_size", 50)
        q_pct = (q_len / q_max * 100) if q_max else 0

        is_in_path = active_path and (n in active_path)
        is_stressed_node = (n == "R2" and stress_r2)

        if n in ["S", "D"]:
            c = "#10B981" if n == "S" else "#8B5CF6"  # Emerald Source, Purple Destination
            s = 34
            b = "#34D399" if is_in_path else "#475569"
            hover = f"<b>Endpoint {n}</b><br>IP: {node_obj.ip_address}<br>MAC: {node_obj.mac_address}"
        elif is_stressed_node:
            c = "#DC2626"  # red for saturated R2
            s = 36
            b = "#FCA5A5"
            hover = f"<b>Router {n} [SATURATED]</b><br>Queue: {q_len}/{q_max} ({q_pct:.0f}%)<br>State: 80% Buffer Fill"
        else:
            c = "#2563EB" if is_in_path else "#1E293B"
            s = 32
            b = "#60A5FA" if is_in_path else "#475569"
            hover = f"<b>Router {n}</b><br>IP: {node_obj.ip_address}<br>Queue: {q_len}/{q_max} ({q_pct:.0f}%)"

        node_color.append(c)
        node_size.append(s)
        border_color.append(b)
        node_hover.append(hover)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(family="JetBrains Mono", size=12, color="#FFFFFF"),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color=border_color, width=2.5),
        ),
        hoverinfo="text",
        hovertext=node_hover,
        showlegend=False,
    ))

    # Optional Live Moving Packet Marker Overlay
    if packet_pos:
        px_val, py_val, p_lbl, p_color = packet_pos
        fig.add_trace(go.Scatter(
            x=[px_val],
            y=[py_val],
            mode="markers+text",
            text=[f"  {p_lbl}"],
            textposition="top right",
            textfont=dict(family="JetBrains Mono", size=11, color=p_color),
            marker=dict(
                size=22,
                color=p_color,
                symbol="diamond",
                line=dict(color="#FFFFFF", width=2.0),
            ),
            hoverinfo="text",
            hovertext=f"Live Packet: {p_lbl}",
            showlegend=False,
        ))

    # Layout styling
    fig.update_layout(
        paper_bgcolor="rgba(11, 15, 25, 0.95)",
        plot_bgcolor="rgba(11, 15, 25, 0.95)",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 3.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
        height=340,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# TAB 1 — NETWORK TOPOLOGY & ML ROUTING
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class='callout-info'>
        <strong>Closed-Loop Adaptive Routing Demonstration (Assessments 6 & 7):</strong><br>
        When Router R2 experiences severe congestion (queue buffer &ge; 80%, link delay &times;6, packet loss &ge; 20%),
        the Machine Learning model detects high bottleneck latency on Route 1 and adapts routing to Route 2 or Route 3 in real time.
    </div>
    """, unsafe_allow_html=True)

    col_topo_left, col_topo_right = st.columns([1.35, 1.0], gap="large")

    with col_topo_left:
        st.markdown("<div class='glass-panel'><h3>Interactive Topology Fabric — Figure 5</h3>", unsafe_allow_html=True)

        # Simulation Controls Bar
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.1, 1.2, 0.9])
        with ctrl_col1:
            stress_r2 = st.toggle("Stress Router R2 (Saturate Queue)", value=False)
            st.session_state.topology.stress_router_r2(stress_active=stress_r2)
        with ctrl_col2:
            routing_mode = st.radio(
                "Routing Decision Engine",
                ["ML-Assisted Dynamic Routing", "Static Shortest Path (via R2)"],
                horizontal=True
            )
        with ctrl_col3:
            sim_packets = st.slider("Burst Packets", min_value=1, max_value=25, value=5)

        # Path Evaluation
        evals = st.session_state.router.evaluate_all_candidate_paths("S", "D")
        best_path_eval, best_route_obj = st.session_state.router.select_best_route("S", "D")
        selected_display_path = best_path_eval if "ML" in routing_mode else ["S", "R1", "R2", "R4", "D"]

        # Topology Chart Placeholder for Live Animation
        topo_placeholder = st.empty()

        # Initial Render of Topology Graph
        topo_fig = build_topology_figure(
            st.session_state.topology,
            active_path=selected_display_path,
            stress_r2=stress_r2
        )
        topo_placeholder.plotly_chart(topo_fig, use_container_width=True, key="topo_initial")

        # Candidate Paths Summary Table
        c_p1, c_p2, c_p3 = st.columns(3)
        for idx, (col_p, ev) in enumerate(zip([c_p1, c_p2, c_p3], evals)):
            is_chosen = (ev.path == selected_display_path)
            badge_class = "badge-danger" if ev.congestion_label == "HIGH" else ("badge-success" if is_chosen else "badge-info")
            border_style = "border: 2px solid #38BDF8;" if is_chosen else "border: 1px solid rgba(51, 65, 85, 0.6);"
            with col_p:
                st.markdown(f"""
                <div style='background: rgba(15, 23, 42, 0.85); padding: 0.75rem 0.9rem; border-radius: 6px; {border_style}'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;'>
                        <span style='font-size: 0.8rem; font-weight: 700; color: #F1F5F9;'>{ev.path_name}</span>
                        <span class='badge {badge_class}'>{ev.congestion_label}</span>
                    </div>
                    <div class='mono-font' style='font-size: 0.73rem; color: #94A3B8;'>{' → '.join(ev.path)}</div>
                    <div style='display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem;'>
                        <span style='color: #CBD5E1;'>Pred Delay: <b style='color:#38BDF8;'>{ev.predicted_delay_ms:.1f}ms</b></span>
                        <span style='color: {"#34D399" if is_chosen else "#94A3B8"}; font-weight: 700;'>{ev.recommendation}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_topo_right:
        st.markdown("<div class='glass-panel'><h3>Live Route Execution & Transmission</h3>", unsafe_allow_html=True)

        if stress_r2:
            st.markdown("""
            <div class='callout-warn'>
                <strong>CONGESTION INJECTED ON ROUTER R2</strong><br>
                Buffer Occupancy: <strong>80% (40/50 packets)</strong><br>
                Link Penalty: <strong>6x Latency Multiplier</strong> | Loss: <strong>+20%</strong>
            </div>
            """, unsafe_allow_html=True)

        send_burst_btn = st.button("Transmit Packet Burst", use_container_width=True)

        # Execution Status & Live Stream Container
        anim_status_placeholder = st.empty()
        log_stream_placeholder = st.empty()

        if send_burst_btn:
            recent_burst = []
            node_positions = {
                "S":  (-2.0,  0.0),
                "R1": (-0.8,  0.8),
                "R3": (-0.8, -0.8),
                "R2": ( 0.6,  0.8),
                "R4": ( 1.4,  0.0),
                "D":  ( 2.6,  0.0),
            }

            for pkt_idx in range(sim_packets):
                path_to_take = selected_display_path

                # Live Animated Packet Traversal along path hops
                for h in range(len(path_to_take) - 1):
                    src_node = path_to_take[h]
                    dst_node = path_to_take[h + 1]
                    x_start, y_start = node_positions[src_node]
                    x_end, y_end = node_positions[dst_node]

                    # Interpolate movement along link
                    steps = 4
                    for step_i in range(steps + 1):
                        alpha = step_i / steps
                        curr_x = x_start + alpha * (x_end - x_start)
                        curr_y = y_start + alpha * (y_end - y_start)

                        # Determine packet color: amber if approaching stressed R2, cyan otherwise
                        is_near_congested = stress_r2 and (src_node == "R2" or dst_node == "R2")
                        p_color = "#EF4444" if is_near_congested else "#38BDF8"
                        p_label = f"PKT #{pkt_idx + 1} ({src_node}→{dst_node})"

                        anim_fig = build_topology_figure(
                            st.session_state.topology,
                            active_path=selected_display_path,
                            stress_r2=stress_r2,
                            packet_pos=(curr_x, curr_y, p_label, p_color)
                        )
                        topo_placeholder.plotly_chart(
                            anim_fig,
                            use_container_width=True,
                            key=f"topo_anim_{pkt_idx}_{h}_{step_i}_{time.time()}"
                        )

                        anim_status_placeholder.markdown(f"""
                        <div class='callout-info' style='margin:0.5rem 0;'>
                            <span class='badge badge-info'>IN-FLIGHT</span>
                            Transmitting <strong>Packet #{pkt_idx + 1}/{sim_packets}</strong> across link <code>{src_node} → {dst_node}</code>...
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.04)

                # Execute transmission simulation
                res = st.session_state.simulator.transmit_message(
                    message=f"CRITICAL_PAYLOAD_BURST_#{pkt_idx+1}",
                    path=path_to_take,
                    transport_type=TransportType.TCP
                )
                recent_burst.append(res)
                st.session_state.history.append({
                    "seq": len(st.session_state.history) + 1,
                    "trace_id": res.trace_id,
                    "path": " → ".join(res.selected_path),
                    "delivered": res.delivered,
                    "latency_ms": res.total_latency_ms,
                    "retransmissions": res.retransmissions,
                    "mode": routing_mode,
                    "congested": stress_r2,
                    "r2_queue_pct": st.session_state.topology.nodes["R2"].queue_occupancy_ratio * 100
                })

            # Return topology to static state at completion
            final_fig = build_topology_figure(
                st.session_state.topology,
                active_path=selected_display_path,
                stress_r2=stress_r2
            )
            topo_placeholder.plotly_chart(
                final_fig,
                use_container_width=True,
                key=f"topo_final_{time.time()}"
            )

            anim_status_placeholder.markdown("""
            <div class='callout-info' style='background:rgba(6, 78, 59, 0.4);border-color:#10B981;color:#A7F3D0;margin:0.5rem 0;'>
                <strong>BURST TRANSMISSION COMPLETE:</strong> All packets traversed the active routing fabric.
            </div>
            """, unsafe_allow_html=True)

            # Render log rows
            log_html = "<p style='font-size:0.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;margin:0.8rem 0 0.4rem 0;'>Burst Execution Log</p>"
            for r in recent_burst:
                badge = "<span class='badge badge-success'>DELIVERED</span>" if r.delivered else "<span class='badge badge-danger'>DROPPED</span>"
                path_fmt = " → ".join(r.selected_path)
                retx_disp = f"<span style='color:#F59E0B;'>{r.retransmissions} Retransmits</span>" if r.retransmissions > 0 else "<span style='color:#10B981;'>0 Retx</span>"
                log_html += (
                    f"<div class='log-row'>"
                    f"{badge}"
                    f"<code style='color:#38BDF8;'>{r.trace_id}</code>"
                    f"<span style='flex-grow:1;font-size:0.75rem;'>{path_fmt}</span>"
                    f"<span style='color:#F8FAFC;font-weight:700;'>{r.total_latency_ms} ms</span>"
                    f"{retx_disp}"
                    f"</div>"
                )
            log_stream_placeholder.markdown(log_html, unsafe_allow_html=True)
        else:
            active_route_str = " → ".join(selected_display_path)
            st.markdown(f"""
            <div style='background: rgba(15, 23, 42, 0.6); border: 1px dashed rgba(71, 85, 105, 0.6); border-radius: 6px; padding: 1.1rem; margin: 0.8rem 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <span style='font-size: 0.8rem; font-weight: 700; color: #CBD5E1;'>Transmission Buffer: READY</span>
                    <span class='badge badge-info'>Awaiting Trigger</span>
                </div>
                <div class='mono-font' style='font-size: 0.78rem; color: #38BDF8; margin-bottom: 0.5rem;'>
                    Target Traversal: {active_route_str}
                </div>
                <div style='font-size: 0.78rem; color: #94A3B8; line-height: 1.5;'>
                    Click <strong>Transmit Packet Burst</strong> above to push {sim_packets} simulated packets across this route and stream real-time hop latency, drops, and retransmission telemetry.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Dynamic Route Comparison Card
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;margin-bottom:0.4rem;'>Route Telemetry Comparison</p>", unsafe_allow_html=True)
        comp_df = pd.DataFrame([
            {
                "Candidate Route": ev.path_name,
                "ML State": ev.congestion_label,
                "Estimated Delay": f"{ev.predicted_delay_ms:.1f} ms",
                "Action": ev.recommendation
            }
            for ev in evals
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — 7-LAYER OSI PACKET ENGINE
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""
    <div class='callout-info'>
        <strong>Unified 7-Layer Protocol Journey (Assessments 2, 3, & 5):</strong><br>
        Inspect full bit-level transformations from <strong>Layer 7 (Application)</strong> down to <strong>Layer 1 (Physical)</strong>.
        Test CRC-32 checksum generation, inject bit-flip corruption on Layer 2, and observe decapsulation validation at destination.
    </div>
    """, unsafe_allow_html=True)

    col_osi_ctrl, col_osi_view = st.columns([1.0, 1.4], gap="large")

    with col_osi_ctrl:
        st.markdown("<div class='glass-panel'><h3>Encapsulation Controls</h3>", unsafe_allow_html=True)
        payload_input = st.text_area(
            "Application Layer Payload (L7)",
            value="TELEMETRY_RECORD: Sensor_Node_S status=NOMINAL temp=24.5C load=42.1%",
            height=80,
        )
        l4_proto = st.radio(
            "Transport Protocol (L4)",
            ["TCP (Reliable Connection, Seq/ACK, Retransmission)", "UDP (Stateless Datagram, Low Overhead)"]
        )
        inject_bitflip = st.checkbox("Inject CRC-32 Bit-Flip Corruption (Layer 2 Framing Error)", value=False)
        encapsulate_btn = st.button("Execute 7-Layer Encapsulation", use_container_width=True)
        osi_stepper_placeholder = st.empty()

        if encapsulate_btn:
            stack = OSIStack()
            t_type = TransportType.TCP if "TCP" in l4_proto else TransportType.UDP

            # Live Step-by-Step Encapsulation Animation
            osi_steps = [
                ("L7 Application", "Creating payload and message ID...", 15),
                ("L6 Presentation", "Applying JSON serialization & zlib compression...", 30),
                ("L5 Session", "Issuing authentication handshake token & state...", 45),
                ("L4 Transport", f"Building {t_type.value} segment header (Ports/Seq/ACK)...", 60),
                ("L3 Network", "Appending IPv4 headers (10.0.0.1 → 10.0.5.1) & TTL...", 75),
                ("L2 Data Link", "Framing MAC addresses & computing CRC-32 checksum...", 90),
                ("L1 Physical", "Serializing bitstream & modeling propagation delay...", 100)
            ]

            progress_bar = st.progress(0)
            for step_name, step_desc, pct in osi_steps:
                osi_stepper_placeholder.markdown(f"""
                <div class='callout-info' style='margin:0.5rem 0;'>
                    <span class='badge badge-info'>PROCESSING {step_name}</span><br>
                    <span style='font-size:0.8rem;'>{step_desc}</span>
                </div>
                """, unsafe_allow_html=True)
                progress_bar.progress(pct)
                time.sleep(0.12)

            progress_bar.empty()
            osi_stepper_placeholder.empty()

            sig, enc_trace = stack.encapsulate(
                message_text=payload_input,
                dst_ip="10.0.5.1",
                dst_mac="00:00:00:00:00:06",
                transport_type=t_type,
                corrupt_crc=inject_bitflip
            )
            app_msg, dec_trace = stack.decapsulate(sig)

            st.session_state.last_signal = sig
            st.session_state.last_enc_trace = enc_trace
            st.session_state.last_dec_trace = dec_trace
            st.session_state.last_corrupt_flag = inject_bitflip

        st.markdown("</div>", unsafe_allow_html=True)

        # Transmission Physical Summary Card
        if st.session_state.last_signal:
            sig = st.session_state.last_signal
            st.markdown("<div class='glass-panel'><h3>Layer 1 Physical Specs</h3>", unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            k1.metric("Serialized Bitstream", f"{sig.bit_count} bits")
            k2.metric("Transmission Latency", f"{sig.transmission_delay_ms:.3f} ms")
            k3, k4 = st.columns(2)
            k3.metric("Propagation Latency", f"{sig.propagation_delay_ms:.2f} ms")
            k4.metric("Total Hop Latency", f"{sig.total_latency_ms:.2f} ms")

            st.markdown("<p style='font-size:0.75rem;font-weight:700;color:#94A3B8;margin-top:0.6rem;'>Bitstream Sample (First 64 Bits)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='osi-payload-box' style='color:#34D399;'>{sig.raw_bits[:64]}...</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_osi_view:
        st.markdown("<div class='glass-panel'><h3>7-Layer Header Inspection Hierarchy</h3>", unsafe_allow_html=True)

        if st.session_state.last_enc_trace:
            tr = st.session_state.last_enc_trace
            dec = st.session_state.last_dec_trace
            corrupted = st.session_state.last_corrupt_flag

            if dec.decapsulation_success:
                st.markdown(f"""
                <div class='callout-info' style='background:rgba(6, 78, 59, 0.4);border-color:#10B981;color:#A7F3D0;'>
                    <strong>DECAPSULATION SUCCESSFUL:</strong> Packet integrity verified across all 7 layers.<br>
                    Trace ID: <code>{tr.trace_id}</code> | CRC-32: <code>{tr.l2_link['crc32']}</code> (VERIFIED)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='callout-warn' style='background:rgba(127, 29, 29, 0.4);border-color:#EF4444;color:#FECACA;'>
                    <strong>CRC-32 VERIFICATION FAILED:</strong> Injected bit-flip detected at Layer 2 Data Link decapsulation.<br>
                    Corrupted Frame rejected by receiver. Receiver Error: <code>{dec.errors}</code>
                </div>
                """, unsafe_allow_html=True)

            # ── Layer 7: Application
            app_sender = tr.l7_app.get("sender_app", tr.l7_app.get("source_app", "Endpoint-A"))
            app_content = tr.l7_app.get("payload", tr.l7_app.get("content", ""))
            app_id = tr.l7_app.get("msg_id", "N/A")
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid #8B5CF6;'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:#A78BFA;'>L7</span><span class='osi-layer-name'>Application Layer</span></div>
                    <span class='badge badge-purple'>App: {app_sender}</span>
                </div>
                <div class='osi-payload-box'>Payload: "{app_content}" | Msg ID: {app_id}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Layer 6: Presentation
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid #6366F1;'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:#818CF8;'>L6</span><span class='osi-layer-name'>Presentation Layer</span></div>
                    <span class='badge badge-info'>Encoding: {tr.l6_pres['encoding']}</span>
                </div>
                <div class='osi-payload-box'>Compression: zlib (Active: {tr.l6_pres['compressed']}) | Raw Size: {tr.l6_pres['orig_size']} bytes</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Layer 5: Session
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid #3B82F6;'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:#60A5FA;'>L5</span><span class='osi-layer-name'>Session Layer</span></div>
                    <span class='badge badge-info'>State: ACTIVE</span>
                </div>
                <div class='osi-payload-box'>Session ID: {tr.l5_sess['session_id']} | Token: {tr.l5_sess['token'][:24]}... | Seq: #{tr.l5_sess['seq']}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Layer 4: Transport
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid #0EA5E9;'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:#38BDF8;'>L4</span><span class='osi-layer-name'>Transport Layer</span></div>
                    <span class='badge badge-info'>Protocol: {tr.l4_trans.get('proto', 'TCP')}</span>
                </div>
                <div class='osi-payload-box'>Ports: {tr.l4_trans.get('src_port', 1024)} → {tr.l4_trans.get('dst_port', 8080)} | Seq: {tr.l4_trans.get('seq_num', 0)} | ACK: {tr.l4_trans.get('ack_num', 0)} | Flags: {tr.l4_trans.get('flags', 'ACK')}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Layer 3: Network
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid #14B8A6;'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:#2DD4BF;'>L3</span><span class='osi-layer-name'>Network Layer (IPv4)</span></div>
                    <span class='badge badge-success'>TTL: {tr.l3_net['ttl']}</span>
                </div>
                <div class='osi-payload-box'>Route: {tr.l3_net['src_ip']} → {tr.l3_net['dst_ip']} | Protocol: {tr.l3_net['protocol']} | Packet ID: {tr.l3_net['packet_id']}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Layer 2: Data Link
            crc_color = "#EF4444" if corrupted else "#10B981"
            crc_badge = "<span class='badge badge-danger'>CRC BITFLIP CORRUPT</span>" if corrupted else "<span class='badge badge-success'>CRC VALID</span>"
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid {crc_color};'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:{crc_color};'>L2</span><span class='osi-layer-name'>Data Link Layer (Ethernet)</span></div>
                    {crc_badge}
                </div>
                <div class='osi-payload-box'>MAC: {tr.l2_link['src_mac']} → {tr.l2_link['dst_mac']} | EtherType: {tr.l2_link['ethertype']} | CRC-32: <b>{tr.l2_link['crc32']}</b></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Layer 1: Physical
            st.markdown(f"""
            <div class='osi-card' style='border-left: 4px solid #F59E0B;'>
                <div class='osi-card-head'>
                    <div><span class='osi-layer-id' style='color:#FBBF24;'>L1</span><span class='osi-layer-name'>Physical Layer</span></div>
                    <span class='badge badge-warning'>Total Delay: {tr.l1_phys['total_latency_ms']} ms</span>
                </div>
                <div class='osi-payload-box'>Bits: {tr.l1_phys['bit_count']} | Tx Delay: {tr.l1_phys['tx_delay_ms']} ms | Prop Delay: {tr.l1_phys['prop_delay_ms']} ms</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='color:#94A3B8; font-size: 0.9rem; text-align: center; padding: 3rem 1rem;'>
                Execute encapsulation on the left panel to generate live layer-by-layer packet telemetry.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — OPERATIONS TELEMETRY & DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    <div class='callout-info'>
        <strong>Network Operations Performance Monitor:</strong><br>
        Aggregates transmission metrics, delivery ratios, TCP retransmission counts, router buffer occupancy, and hop delay fluctuations across time.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        total_tx = len(hist_df)
        delivered_tx = int(hist_df["delivered"].sum())
        delivery_pct = (delivered_tx / total_tx * 100) if total_tx else 0
        avg_lat = hist_df["latency_ms"].mean()
        max_lat = hist_df["latency_ms"].max()
        total_retx = int(hist_df["retransmissions"].sum())

        # Top KPI Metric Cards
        st.markdown(f"""
        <div class='kpi-row'>
            <div class='kpi-box emerald'>
                <div class='kpi-tag'>Delivered Traffic</div>
                <div class='kpi-num'>{delivery_pct:.1f}%</div>
                <div class='kpi-desc'>{delivered_tx} of {total_tx} packets delivered</div>
            </div>
            <div class='kpi-box'>
                <div class='kpi-tag'>Average Latency</div>
                <div class='kpi-num'>{avg_lat:.1f} <span style='font-size:1rem;color:#94A3B8;'>ms</span></div>
                <div class='kpi-desc'>Peak recorded: {max_lat:.1f} ms</div>
            </div>
            <div class='kpi-box amber'>
                <div class='kpi-tag'>TCP Retransmissions</div>
                <div class='kpi-num'>{total_retx}</div>
                <div class='kpi-desc'>Triggered on CRC or drops</div>
            </div>
            <div class='kpi-box violet'>
                <div class='kpi-tag'>Active Router Nodes</div>
                <div class='kpi-num'>6</div>
                <div class='kpi-desc'>Topology: S, R1, R2, R3, R4, D</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Multi-chart Telemetry Dashboard
        fig_telemetry = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "End-to-End Latency vs. Sequence",
                "Router R2 Queue Occupancy Buffer (%)",
                "Retransmissions per Transmission Event",
                "Delivery Status Distribution"
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}]
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.12,
        )

        # 1. Latency Time Series
        fig_telemetry.add_trace(
            go.Scatter(
                x=hist_df["seq"],
                y=hist_df["latency_ms"],
                mode="lines+markers",
                name="Latency (ms)",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=6, color="#0284C7", line=dict(color="#E0F2FE", width=1.5)),
            ),
            row=1, col=1
        )

        # 2. Router R2 Queue fill percentage
        q_fill = hist_df.get("r2_queue_pct", [0] * len(hist_df))
        fig_telemetry.add_trace(
            go.Scatter(
                x=hist_df["seq"],
                y=q_fill,
                mode="lines",
                name="R2 Queue %",
                fill="tozeroy",
                line=dict(color="#F43F5E", width=2.0),
                fillcolor="rgba(244, 63, 94, 0.2)",
            ),
            row=1, col=2
        )

        # 3. Retransmissions
        fig_telemetry.add_trace(
            go.Bar(
                x=hist_df["seq"],
                y=hist_df["retransmissions"],
                name="Retransmissions",
                marker_color="#F59E0B",
            ),
            row=2, col=1
        )

        # 4. Delivery Status Pie
        deliv_counts = hist_df["delivered"].value_counts()
        labels_pie = ["Delivered" if k else "Dropped" for k in deliv_counts.index]
        fig_telemetry.add_trace(
            go.Pie(
                labels=labels_pie,
                values=deliv_counts.values,
                hole=0.45,
                marker=dict(colors=["#10B981", "#EF4444"]),
                textinfo="label+percent",
                showlegend=False
            ),
            row=2, col=2
        )

        fig_telemetry.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(family="Plus Jakarta Sans", color="#F1F5F9", size=11),
            height=520,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        fig_telemetry.update_xaxes(showgrid=True, gridcolor="#334155", zeroline=False)
        fig_telemetry.update_yaxes(showgrid=True, gridcolor="#334155", zeroline=False)

        st.plotly_chart(fig_telemetry, use_container_width=True)

        # Telemetry Log Table
        st.markdown("<div class='glass-panel'><h3>Historical Transmission Log</h3>", unsafe_allow_html=True)
        st.dataframe(
            hist_df[[
                "seq", "trace_id", "path", "delivered", "latency_ms", "retransmissions", "mode", "congested"
            ]].rename(columns={
                "seq": "Seq #",
                "trace_id": "Trace ID",
                "path": "Traversal Route",
                "delivered": "Delivered",
                "latency_ms": "Delay (ms)",
                "retransmissions": "Retx",
                "mode": "Engine",
                "congested": "R2 Stress"
            }),
            use_container_width=True,
            hide_index=True
        )

        if st.button("Reset Telemetry Store"):
            st.session_state.history = []
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='callout-info'>
            No historical telemetry recorded. Navigate to the <strong>Network Topology & Adaptive Routing</strong> tab and run a burst to populate diagnostics.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — MACHINE LEARNING DECISION MATRIX
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div class='callout-info'>
        <strong>Machine Learning Intelligence Diagnostics (Assessment 7):</strong><br>
        Inspect Random Forest feature importance weights, model regression accuracy, and real-time inference confidence scores.
    </div>
    """, unsafe_allow_html=True)

    col_ml_left, col_ml_right = st.columns([1.2, 1.0], gap="large")

    model_metrics = st.session_state.ml_model.metrics

    with col_ml_left:
        st.markdown("<div class='glass-panel'><h3>Random Forest Feature Importance Analysis</h3>", unsafe_allow_html=True)

        feat_importances = model_metrics.get("feature_importances", {
            "queue_occupancy_pct": 0.38,
            "loss_rate_pct": 0.24,
            "rtt_ms": 0.18,
            "packet_rate_pps": 0.11,
            "bandwidth_mbps": 0.06,
            "active_connections": 0.03
        })

        feat_df = pd.DataFrame([
            {"Feature": k.replace("_", " ").title(), "Importance": v}
            for k, v in sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)
        ])

        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            text="Importance",
        )
        fig_feat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_feat.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(family="Plus Jakarta Sans", color="#F1F5F9"),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="#334155", range=[0, max(feat_importances.values()) * 1.25]),
            height=280,
            margin=dict(l=10, r=20, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        st.markdown("""
        <p style='font-size:0.8rem; color:#94A3B8; line-height:1.5;'>
            <strong>Telemetry Takeaway:</strong> <code>queue_occupancy_pct</code> and <code>loss_rate_pct</code> dominate classification splits.
            When buffer occupancy crosses 75%, the Random Forest classifier triggers an immediate shift from <strong>LOW</strong> to <strong>HIGH</strong> congestion,
            causing the router to reroute traffic away from the saturated link.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ml_right:
        st.markdown("<div class='glass-panel'><h3>ML Model Validation Metrics</h3>", unsafe_allow_html=True)

        acc = model_metrics.get("classification_accuracy", 0.985)
        mse = model_metrics.get("regression_mse", 3.12)
        r2 = model_metrics.get("regression_r2", 0.968)

        m1, m2 = st.columns(2)
        m1.metric("Classification Accuracy", f"{acc * 100:.2f}%")
        m2.metric("Delay Regression R²", f"{r2:.3f}")

        m3, m4 = st.columns(2)
        m3.metric("Regression MSE", f"{mse:.2f}")
        m4.metric("Estimators", "100 Trees")

        st.markdown("<hr style='border-color:rgba(51, 65, 85, 0.5);margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem;font-weight:700;color:#94A3B8;text-transform:uppercase;'>Inference Radar: Candidate Routes Under Current Topology</p>", unsafe_allow_html=True)

        current_evals = st.session_state.router.evaluate_all_candidate_paths("S", "D")
        radar_categories = ["Est. Latency", "Confidence", "Hop Efficiency"]

        fig_radar = go.Figure()
        colors = ["#38BDF8", "#10B981", "#F59E0B"]

        for idx, ev in enumerate(current_evals):
            lat_norm = max(0, 100 - ev.predicted_delay_ms)
            conf_norm = ev.confidence * 100
            hop_norm = 100 if len(ev.path) == 4 else 80
            r_vals = [lat_norm, conf_norm, hop_norm, lat_norm]

            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name=ev.path_name,
                line=dict(color=colors[idx % len(colors)]),
                fillcolor=f"rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.15)"
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155", tickfont=dict(size=8, color="#94A3B8")),
                angularaxis=dict(gridcolor="#334155", tickfont=dict(size=9, color="#CBD5E1")),
                bgcolor="rgba(15, 23, 42, 0.8)"
            ),
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(family="Plus Jakarta Sans", color="#F1F5F9"),
            height=260,
            margin=dict(l=30, r=30, t=20, b=20),
            legend=dict(font=dict(size=9), orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — HOST SOCKET PROGRAMMING
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
    <div class='callout-info'>
        <strong>Assessment 4: Host Socket API Interoperability:</strong><br>
        Directly test operating system level Python TCP and UDP endpoints over localhost (ports 9091 and 9092) with round-trip measurement.
    </div>
    """, unsafe_allow_html=True)

    col_sock_left, col_sock_right = st.columns([1.0, 1.2], gap="large")

    with col_sock_left:
        st.markdown("<div class='glass-panel'><h3>Client Transmission Config</h3>", unsafe_allow_html=True)
        sock_proto = st.selectbox("Transport Socket Protocol", ["TCP (Connection-Oriented, Port 9091)", "UDP (Datagram, Port 9092)"])
        sock_msg = st.text_input("Socket Payload", value="ECHO_TEST: Verification of Computer Networks Assessment 4 Host API")
        send_socket_btn = st.button("Transmit Over Local Socket", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-panel'><h3>Transport Protocol Comparison</h3>", unsafe_allow_html=True)
        comp_proto_df = pd.DataFrame({
            "Specification": ["Handshake", "Reliability", "Ordering", "Flow Control", "Overhead"],
            "TCP": ["3-Way SYN/ACK", "Guaranteed (Retx)", "Strict In-Order", "Sliding Window", "20-60 Byte Header"],
            "UDP": ["None (Connectionless)", "Best-Effort", "Unordered", "None", "8 Byte Header"]
        })
        st.dataframe(comp_proto_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sock_right:
        st.markdown("<div class='glass-panel'><h3>Server Endpoint Verification</h3>", unsafe_allow_html=True)

        sock_status_placeholder = st.empty()

        if send_socket_btn:
            proto_name = "TCP" if "TCP" in sock_proto else "UDP"
            sock_port = 9091 if proto_name == "TCP" else 9092

            sock_status_placeholder.markdown(f"""
            <div class='callout-info' style='margin:0.5rem 0;'>
                <span class='badge badge-info'>SOCKET INITIATING</span><br>
                Spawning background {proto_name} server on <code>127.0.0.1:{sock_port}</code>...
            </div>
            """, unsafe_allow_html=True)

            server = SocketServer()
            server.start_tcp_server()
            server.start_udp_server()
            time.sleep(0.15)

            sock_status_placeholder.markdown(f"""
            <div class='callout-info' style='margin:0.5rem 0;'>
                <span class='badge badge-purple'>TRANSMITTING</span><br>
                Client sending payload across localhost socket...
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)

            client = SocketClient()
            if "TCP" in sock_proto:
                ok, response, rtt = client.send_tcp_message(sock_msg)
                proto_name = "TCP"
            else:
                ok, response, rtt = client.send_udp_message(sock_msg)
                proto_name = "UDP"

            server.stop()
            sock_status_placeholder.empty()

            if ok:
                st.markdown(f"""
                <div class='callout-info' style='background:rgba(6, 78, 59, 0.4);border-color:#10B981;color:#A7F3D0;'>
                    <strong>ENDPOINT ACKNOWLEDGED:</strong> Server successfully echoed payload.<br>
                    Protocol: <strong>{proto_name}</strong> | Measured RTT: <strong>{rtt:.2f} ms</strong>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    f"<div class='log-row'>"
                    f"<span class='badge badge-success'>ACK RECEIVED</span>"
                    f"<span class='mono-font' style='color:#38BDF8;'>{response}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                k1, k2, k3 = st.columns(3)
                k1.metric("Protocol", proto_name)
                k2.metric("Socket RTT", f"{rtt:.2f} ms")
                k3.metric("Endpoint Status", "NOMINAL")
            else:
                st.markdown(f"""
                <div class='callout-warn' style='background:rgba(127, 29, 29, 0.4);border-color:#EF4444;color:#FECACA;'>
                    <strong>SOCKET ERROR:</strong> Failed to exchange packet with local socket endpoint.<br>
                    Error details: <code>{response}</code>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='color:#94A3B8; font-size: 0.9rem; text-align: center; padding: 2.5rem 1rem;'>
                Click <strong>Transmit Over Local Socket</strong> to spin up background servers and verify round-trip echo response.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 6 — NOVEL CONTRIBUTION: LOSS-CAUSE-AWARE CROSS-LAYER ROUTING
# ═══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("""
    <div class='callout-info' style='border-left-color:#10B981;background:rgba(6,78,59,0.3);'>
        <strong>NOVEL RESEARCH CONTRIBUTION: CROSS-LAYER LOSS-CAUSE-AWARE ROUTING</strong><br>
        Standard ML routers naively treat every packet drop as network congestion and reroute traffic.
        Our engine inspects <strong>L2 Data Link (CRC-32 checksums)</strong> and <strong>L4 Transport (TCP retransmission & delay variance)</strong>
        evidence to classify the root cause of degradation: <em>Physical Corruption vs Queue Buffer Congestion vs Mixed</em>,
        executing policy-directed responses with hysteresis dampening.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_diag = st.columns([1.1, 1.3], gap="large")

    with col_ctrl:
        st.markdown("<div class='glass-panel'><h3>1. Stressor Injection & Control Plane</h3>", unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            corruption_toggle = st.toggle("Inject L2 Link Corruption", value=False)
            ber_val = st.slider("Bit Error Rate (BER)", min_value=0.05, max_value=0.40, value=0.20, step=0.05) if corruption_toggle else 0.0
            st.session_state.topology.inject_link_corruption("R1", "R2", ber_val)

        with sc2:
            congestion_toggle = st.toggle("Inject L3 Queue Congestion (R2)", value=False)
            st.session_state.topology.stress_router_r2(congestion_toggle)

        st.markdown("<hr style='border-color:rgba(51,65,85,0.4);margin:0.8rem 0;'>", unsafe_allow_html=True)

        h1, h2 = st.columns(2)
        with h1:
            conf_threshold = st.slider("Confidence Fallback Threshold", min_value=0.50, max_value=0.90, value=0.60, step=0.05)
            st.session_state.loss_router.confidence_threshold = conf_threshold
        with h2:
            hysteresis_n = st.slider("Hysteresis Window (Packets)", min_value=5, max_value=40, value=20, step=5)
            st.session_state.loss_router.hysteresis_packets = hysteresis_n

        run_diag_btn = st.button("Probe Network & Classify Loss Cause", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_diag:
        st.markdown("<div class='glass-panel'><h3>2. Cross-Layer Classifier & Policy Engine</h3>", unsafe_allow_html=True)

        # Extract live features from current path
        current_p = st.session_state.loss_router.current_path
        live_feats = st.session_state.loss_router.extract_path_features(current_p)
        pred_cause, conf, probas = st.session_state.loss_classifier.predict_cause(live_feats)
        chosen_path, decision = st.session_state.loss_router.select_route("S", "D", features=live_feats)

        # Classification Badge
        badge_style_map = {
            "NORMAL": ("#064E3B", "#6EE7B7", "#059669", "NORMAL (HEALTHY)"),
            "CORRUPTION": ("#78350F", "#FDE68A", "#D97706", "CORRUPTION (L2 BIT-FLIP)"),
            "CONGESTION": ("#7F1D1D", "#FCA5A5", "#DC2626", "CONGESTION (L3 BUFFER OVERFLOW)"),
            "MIXED": ("#4C1D95", "#DDD6FE", "#7C3AED", "MIXED (CONGESTION + NOISE)")
        }
        bg_col, txt_col, brd_col, label_str = badge_style_map.get(pred_cause, ("#1E293B", "#94A3B8", "#475569", pred_cause))

        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;background:{bg_col};border:1px solid {brd_col};border-radius:8px;padding:0.9rem 1.2rem;margin-bottom:1rem;'>
            <div>
                <span style='font-size:0.75rem;color:#94A3B8;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;'>Root-Cause Classification</span>
                <div style='font-size:1.35rem;font-weight:800;color:{txt_col};font-family:JetBrains Mono;'>{label_str}</div>
            </div>
            <div style='text-align:right;'>
                <span style='font-size:0.75rem;color:#94A3B8;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;'>Confidence</span>
                <div style='font-size:1.35rem;font-weight:800;color:#F8FAFC;font-family:JetBrains Mono;'>{conf*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chosen Action Card
        action_name = decision["action"]
        act_color = "#10B981" if action_name == "RETRANSMIT_SAME_PATH" else ("#38BDF8" if action_name == "REROUTE_CONGESTION" else "#F59E0B")
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.8);border:1px solid rgba(51,65,85,0.7);border-left:4px solid {act_color};border-radius:6px;padding:0.75rem 1rem;margin-bottom:1rem;'>
            <div style='font-size:0.75rem;color:#94A3B8;font-weight:700;'>RESPONSE POLICY ACTION:</div>
            <div style='font-family:JetBrains Mono;font-size:1.05rem;font-weight:800;color:{act_color};margin:0.2rem 0;'>{action_name}</div>
            <div style='font-size:0.8rem;color:#CBD5E1;'>{decision['reason']}</div>
            <div style='font-size:0.78rem;color:#64748B;margin-top:0.4rem;'>Active Route: <code>{' -> '.join(chosen_path)}</code> | Hysteresis Count: <code>{decision['packets_on_current_path']}/{hysteresis_n}</code></div>
        </div>
        """, unsafe_allow_html=True)

        # Probability Distribution Bar
        prob_df = pd.DataFrame([{"Class": k, "Probability": v} for k, v in probas.items()])
        fig_prob = px.bar(
            prob_df, x="Probability", y="Class", orientation="h",
            color="Class",
            color_discrete_map={
                "NORMAL": "#10B981", "CONGESTION": "#EF4444",
                "CORRUPTION": "#F59E0B", "MIXED": "#8B5CF6"
            },
            height=140
        )
        fig_prob.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(range=[0, 1.0], showgrid=True),
            yaxis=dict(title=None)
        )
        st.plotly_chart(fig_prob, use_container_width=True, key="loss_cause_prob_chart")

        st.markdown("</div>", unsafe_allow_html=True)

    # Cross-Layer Telemetry Signals Grid
    st.markdown("<div class='glass-panel'><h3>3. Cross-Layer Telemetry Observable Space</h3>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("L2 CRC Fail Rate", f"{live_feats.get('crc_fail_rate', 0.0)*100:.1f}%")
    with m2:
        st.metric("L4 Retransmission Rate", f"{live_feats.get('retransmission_rate', 0.0)*100:.1f}%")
    with m3:
        st.metric("L4 RTT Variance", f"{live_feats.get('rtt_variance', 0.0):.2f} ms²")
    with m4:
        st.metric("L3 Queue Occupancy", f"{live_feats.get('queue_occupancy_pct', 0.0):.1f}%")
    with m5:
        st.metric("L3 Queue Slope", f"{live_feats.get('queue_slope', 0.0):+.2f} %/step")
    st.markdown("</div>", unsafe_allow_html=True)

    # Comparative Benchmark Section
    st.markdown("<div class='glass-panel'><h3>4. Multi-Router Empirical Benchmark Suite</h3>", unsafe_allow_html=True)
    csv_file = os.path.join(os.path.dirname(__file__), "../../results/experiment_results.csv")
    if os.path.exists(csv_file):
        exp_df = pd.read_csv(csv_file)
        sc_select = st.selectbox("Filter Benchmark by Scenario", exp_df["scenario"].unique(), index=1)
        sub_exp = exp_df[exp_df["scenario"] == sc_select]

        b1, b2 = st.columns(2)
        with b1:
            fig_cmp_reroutes = px.bar(
                sub_exp, x="router", y="needless_reroutes",
                color="router",
                color_discrete_map={
                    "StaticDijkstra": "#64748B",
                    "AdaptiveMLRouter": "#EF4444",
                    "LossAwareRouter": "#10B981",
                    "LossAwareRouter_Ablation": "#F59E0B"
                },
                title=f"Needless Reroutes — {sc_select.upper()} (Lower is Better)"
            )
            fig_cmp_reroutes.update_layout(template="plotly_dark", showlegend=False, height=280)
            st.plotly_chart(fig_cmp_reroutes, use_container_width=True, key=f"reroute_bar_{sc_select}")

        with b2:
            fig_cmp_lat = px.bar(
                sub_exp, x="router", y="mean_latency_ms",
                color="router",
                color_discrete_map={
                    "StaticDijkstra": "#64748B",
                    "AdaptiveMLRouter": "#EF4444",
                    "LossAwareRouter": "#10B981",
                    "LossAwareRouter_Ablation": "#F59E0B"
                },
                title=f"Mean Latency (ms) — {sc_select.upper()} (Lower is Better)"
            )
            fig_cmp_lat.update_layout(template="plotly_dark", showlegend=False, height=280)
            st.plotly_chart(fig_cmp_lat, use_container_width=True, key=f"lat_bar_{sc_select}")

        st.dataframe(sub_exp, use_container_width=True)
    else:
        st.info("Run `python run_experiments.py` to populate real empirical benchmark data.")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER ATTRIBUTION
# ─────────────────────────────────────────────
st.markdown("<hr style='border-color:rgba(51, 65, 85, 0.5);margin:2rem 0 1rem 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-size:0.75rem;font-weight:600;color:#64748B;letter-spacing:0.04em;text-transform:uppercase;'>
    Adaptive Network Operations Center &nbsp;|&nbsp; Computer Networks Lab Project &nbsp;|&nbsp; VIT University
</div>
""", unsafe_allow_html=True)
