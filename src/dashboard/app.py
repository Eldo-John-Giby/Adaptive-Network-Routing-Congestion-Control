"""
Interactive Visual Dashboard for Adaptive Network Routing & Congestion Control
Built with Streamlit and Matplotlib.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.osi.packet_engine import OSIStack
from src.osi.l4_transport import TransportType
from src.network.topology import NetworkTopology
from src.network.simulator import NetworkSimulator
from src.ml.model import NetworkCongestionModel
from src.ml.controller import AdaptiveMLRouter
from src.endpoints.socket_server import SocketServer
from src.endpoints.socket_client import SocketClient

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Network Routing — ML Congestion Control",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# HIGH-CONTRAST THEME STYLES (DARK/LIGHT ADAPTIVE)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1280px; }

    /* Theme-adaptive CSS variables */
    :root {
        --bg-panel: rgba(15, 23, 42, 0.04);
        --border-panel: rgba(148, 163, 184, 0.35);
        --text-primary: #0F172A;
        --text-secondary: #334155;
        --text-muted: #64748B;
        --accent-blue: #2563EB;
        --accent-blue-hover: #1D4ED8;
        --card-bg: #FFFFFF;
        --banner-info-bg: #EFF6FF;
        --banner-info-border: #3B82F6;
        --banner-info-text: #1E3A8A;
        --banner-warn-bg: #FEF3C7;
        --banner-warn-border: #F59E0B;
        --banner-warn-text: #78350F;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-panel: rgba(30, 41, 59, 0.7);
            --border-panel: rgba(148, 163, 184, 0.3);
            --text-primary: #F8FAFC;
            --text-secondary: #E2E8F0;
            --text-muted: #94A3B8;
            --accent-blue: #3B82F6;
            --accent-blue-hover: #60A5FA;
            --card-bg: #1E293B;
            --banner-info-bg: rgba(30, 58, 138, 0.4);
            --banner-info-border: #60A5FA;
            --banner-info-text: #DBEAFE;
            --banner-warn-bg: rgba(120, 53, 15, 0.4);
            --banner-warn-border: #FBBF24;
            --banner-warn-text: #FEF3C7;
        }
    }

    /* Page header */
    .page-header {
        border-left: 5px solid var(--accent-blue);
        padding: 0.5rem 0 0.5rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .page-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.02em;
    }
    .page-header p {
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin: 0.3rem 0 0 0;
    }

    /* Section label */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-blue);
        margin-bottom: 0.5rem;
    }

    /* Panel container */
    .panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.1rem;
    }
    .panel h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.9rem 0;
    }

    /* High-contrast result rows */
    .result-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.65rem 1rem;
        border-radius: 6px;
        background: var(--card-bg);
        border: 1px solid var(--border-panel);
        margin-bottom: 0.5rem;
        font-size: 0.86rem;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .result-row code {
        color: #38BDF8;
        font-weight: 600;
    }

    /* High-contrast status badges */
    .badge-ok {
        background: #14532D;
        color: #BBF7D0;
        border: 1px solid #22C55E;
        border-radius: 4px;
        padding: 3px 9px;
        font-size: 0.75rem;
        font-weight: 700;
        white-space: nowrap;
        letter-spacing: 0.04em;
    }
    .badge-fail {
        background: #7F1D1D;
        color: #FECACA;
        border: 1px solid #EF4444;
        border-radius: 4px;
        padding: 3px 9px;
        font-size: 0.75rem;
        font-weight: 700;
        white-space: nowrap;
        letter-spacing: 0.04em;
    }
    .badge-warn {
        background: #78350F;
        color: #FEF08A;
        border: 1px solid #F59E0B;
        border-radius: 4px;
        padding: 3px 9px;
        font-size: 0.75rem;
        font-weight: 700;
        white-space: nowrap;
        letter-spacing: 0.04em;
    }

    /* KPI metric cards */
    .kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        flex: 1;
        background: var(--card-bg);
        border: 1px solid var(--border-panel);
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }
    .kpi-sub { font-size: 0.8rem; font-weight: 500; color: var(--text-muted); margin-top: 0.3rem; }

    .divider { border: none; border-top: 1px solid var(--border-panel); margin: 1.3rem 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid var(--border-panel);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-muted);
        padding: 0.65rem 1.3rem;
        border-radius: 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        border-bottom: 3px solid var(--accent-blue);
        background: transparent;
    }

    /* Terminal-style high-contrast topology */
    .topo-block {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.86rem;
        font-weight: 500;
        background: #0B0F19;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        line-height: 1.8;
        white-space: pre;
    }
    .topo-node { color: #60A5FA; font-weight: 700; }
    .topo-link { color: #34D399; font-weight: 600; }

    /* High-contrast notification banners */
    .info-banner {
        background: var(--banner-info-bg);
        border: 1px solid var(--banner-info-border);
        border-left: 4px solid var(--banner-info-border);
        border-radius: 6px;
        padding: 0.8rem 1.1rem;
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--banner-info-text);
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .warn-banner {
        background: var(--banner-warn-bg);
        border: 1px solid var(--banner-warn-border);
        border-left: 4px solid var(--banner-warn-border);
        border-radius: 6px;
        padding: 0.8rem 1.1rem;
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--banner-warn-text);
        margin-bottom: 1rem;
        line-height: 1.5;
    }

    /* High-contrast buttons */
    .stButton > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.55rem 1.3rem;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background: #1D4ED8 !important;
        border-color: #1E40AF !important;
        color: #FFFFFF !important;
    }

    /* Expanders header contrast */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--text-primary) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
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
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Adaptive Network Routing &amp; Congestion Control</h1>
    <p>End-to-End 7-Layer OSI Simulation &nbsp;|&nbsp; Router Queue Dynamics &nbsp;|&nbsp; ML-Driven Route Optimization</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "OSI Stack Inspector",
    "Network Topology & ML Routing",
    "Performance Telemetry",
    "Socket Programming",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — OSI STACK INSPECTOR
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<p class='section-label'>Assessment 2 &amp; 5 — Protocol Stack Demonstration</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-banner'>
        Configure a payload below and run the encapsulation engine to inspect every header added as data travels from
        <strong>Layer 7 (Application)</strong> down to <strong>Layer 1 (Physical)</strong>, then verify decapsulation at the receiver.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='panel'><h3>Transmission Configuration</h3>", unsafe_allow_html=True)
        input_msg = st.text_area(
            "Application Layer Payload (L7)",
            value="CRITICAL_PAYLOAD: Sensor readings telemetry stream #1042.",
            height=90,
        )
        proto_choice = st.radio(
            "Transport Protocol (L4)",
            ["TCP  —  Reliable, Seq/ACK, Retransmission", "UDP  —  Unreliable, Low Latency"],
        )
        corrupt_crc = st.checkbox(
            "Inject CRC Frame Corruption  (bit-flip on L2 Data Link)",
            value=False,
        )
        send_btn = st.button("Run Encapsulation & Inspect Stack", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='font-size:1rem;font-weight:600;margin-bottom:0.8rem;'>Layer-by-Layer Header Inspection</h3>", unsafe_allow_html=True)
        if send_btn:
            stack = OSIStack()
            t_type = TransportType.TCP if "TCP" in proto_choice else TransportType.UDP
            sig, enc_trace = stack.encapsulate(input_msg, transport_type=t_type, corrupt_crc=corrupt_crc)
            app_msg, dec_trace = stack.decapsulate(sig)

            if dec_trace.decapsulation_success:
                st.markdown(
                    f"<div class='info-banner'>Packet encapsulated and delivered. Trace ID: <strong>{enc_trace.trace_id}</strong></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='warn-banner'>Decapsulation failed — CRC mismatch detected at receiver. Trace ID: <strong>{enc_trace.trace_id}</strong></div>",
                    unsafe_allow_html=True,
                )

            with st.expander("L7  —  Application Layer", expanded=True):
                st.json(enc_trace.l7_app)
            with st.expander("L6  —  Presentation Layer  (encoding & compression)", expanded=False):
                st.json(enc_trace.l6_pres)
            with st.expander("L5  —  Session Layer  (token & state)", expanded=False):
                st.json(enc_trace.l5_sess)
            with st.expander("L4  —  Transport Layer  (TCP/UDP headers & checksum)", expanded=True):
                st.json(enc_trace.l4_trans)
            with st.expander("L3  —  Network Layer  (IPv4 header & TTL)", expanded=False):
                st.json(enc_trace.l3_net)
            with st.expander("L2  —  Data Link Layer  (Ethernet frame & CRC-32)", expanded=True):
                st.json(enc_trace.l2_link)
                if corrupt_crc:
                    st.markdown(
                        "<div class='warn-banner'>CRC-32 Verification FAILED — corrupted frame caught by receiver.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='info-banner'>CRC-32 Verification PASSED — frame integrity confirmed.</div>",
                        unsafe_allow_html=True,
                    )
            with st.expander("L1  —  Physical Layer  (bitstream & delay)", expanded=False):
                st.json(enc_trace.l1_phys)
        else:
            st.markdown(
                "<div style='color:var(--text-muted);font-size:0.9rem;padding-top:2rem;text-align:center;font-weight:500;'>"
                "Configure transmission parameters and click <strong>Run Encapsulation &amp; Inspect Stack</strong> to begin."
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — NETWORK TOPOLOGY & ML ROUTING
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<p class='section-label'>Assessment 6 &amp; 7 — Routing Protocol &amp; Congestion Control</p>", unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([1, 1], gap="large")

    with top_col1:
        st.markdown("<div class='panel'><h3>Network Topology  —  Figure 5</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='topo-block'>  <span class='topo-node'>[S]</span> ──(8ms)──► <span class='topo-node'>[R1]</span> ──(5ms)──► <span class='topo-node'>[R2]</span> ──(4ms)──► <span class='topo-node'>[R4]</span> ──(7ms)──► <span class='topo-node'>[D]</span>
   │              │                              ▲       ▲
   └──(14ms)──►  <span class='topo-node'>[R3]</span> ────────────(3ms)─────────┘       │
                  └──────────────────────────────────────┘</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        topo_data = {
            "Route": ["Route 1", "Route 2", "Route 3"],
            "Path": ["S → R1 → R2 → R4 → D", "S → R1 → R3 → R4 → D", "S → R3 → R4 → D"],
            "Base Latency": ["24 ms", "24 ms", "24 ms"],
        }
        st.dataframe(pd.DataFrame(topo_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'><h3>Simulation Controls</h3>", unsafe_allow_html=True)
        stress_r2 = st.checkbox(
            "Congest Router R2  (queue overload · 6x latency · 20% packet loss)",
            value=False,
        )
        st.session_state.topology.stress_router_r2(stress_active=stress_r2)

        if stress_r2:
            st.markdown(
                "<div class='warn-banner'>R2 is congested. Queue filled to 80%. Link delay multiplied 6x. Packet loss rate +20%.</div>",
                unsafe_allow_html=True,
            )

        routing_mode = st.radio(
            "Routing Decision Policy",
            ["ML-Assisted Adaptive Routing", "Static Routing  (fixed path via R2)"],
        )
        sim_packets = st.slider("Packets to Transmit", min_value=1, max_value=20, value=5)
        run_sim_btn = st.button("Run Simulation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_col2:
        st.markdown("<div class='panel'><h3>ML Route Predictions</h3>", unsafe_allow_html=True)
        evals = st.session_state.router.evaluate_all_candidate_paths("S", "D")
        eval_records = []
        for ev in evals:
            eval_records.append({
                "Route": ev.path_name,
                "Path": " → ".join(ev.path),
                "Base Delay": f"{ev.actual_base_delay_ms} ms",
                "Congestion State": ev.congestion_label,
                "Confidence": f"{ev.confidence * 100:.1f}%",
                "Predicted Delay": f"{ev.predicted_delay_ms} ms",
                "Recommendation": ev.recommendation,
            })
        st.dataframe(pd.DataFrame(eval_records), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_sim_btn:
            st.markdown("<div class='panel'><h3>Transmission Results</h3>", unsafe_allow_html=True)
            results = []
            for i in range(sim_packets):
                if "ML" in routing_mode:
                    chosen_path, _ = st.session_state.router.select_best_route("S", "D")
                else:
                    chosen_path = ["S", "R1", "R2", "R4", "D"]

                res = st.session_state.simulator.transmit_message(
                    message=f"SIM_MSG_{i + 1}",
                    path=chosen_path,
                    transport_type=TransportType.TCP,
                )
                results.append(res)
                st.session_state.history.append({
                    "seq": len(st.session_state.history) + 1,
                    "path": " → ".join(res.selected_path),
                    "delivered": res.delivered,
                    "latency_ms": res.total_latency_ms,
                    "retransmissions": res.retransmissions,
                    "mode": routing_mode,
                    "congested": stress_r2,
                })

            for r in results:
                badge = "<span class='badge-ok'>DELIVERED</span>" if r.delivered else "<span class='badge-fail'>DROPPED</span>"
                path_str = " → ".join(r.selected_path)
                st.markdown(
                    f"<div class='result-row'>{badge}"
                    f"&nbsp; <code>{r.trace_id}</code>"
                    f"&nbsp;·&nbsp; <strong>{path_str}</strong>"
                    f"&nbsp;·&nbsp; <strong>{r.total_latency_ms} ms</strong>"
                    f"&nbsp;·&nbsp; {r.retransmissions} retx</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE TELEMETRY
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<p class='section-label'>Live Performance Metrics &amp; Telemetry History</p>", unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        delivered_count = hist_df["delivered"].sum()
        total_count = len(hist_df)
        delivery_rate = (delivered_count / total_count * 100) if total_count else 0
        avg_latency = hist_df["latency_ms"].mean()
        total_retx = int(hist_df["retransmissions"].sum())

        st.markdown(f"""
        <div class='kpi-grid'>
            <div class='kpi-card'>
                <div class='kpi-label'>Packets Sent</div>
                <div class='kpi-value'>{total_count}</div>
                <div class='kpi-sub'>total transmissions</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-label'>Delivery Rate</div>
                <div class='kpi-value'>{delivery_rate:.1f}%</div>
                <div class='kpi-sub'>{int(delivered_count)} of {total_count} delivered</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-label'>Avg End-to-End Latency</div>
                <div class='kpi-value'>{avg_latency:.1f} ms</div>
                <div class='kpi-sub'>across all packets</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-label'>TCP Retransmissions</div>
                <div class='kpi-value'>{total_retx}</div>
                <div class='kpi-sub'>total retransmit events</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── High-Contrast Matplotlib Visualizations ──
        chart_bg = "#0F172A"
        chart_fg = "#F8FAFC"
        grid_color = "#334155"

        mpl.rcParams.update({
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": grid_color,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.labelcolor": "#CBD5E1",
            "axes.titlecolor": "#F8FAFC",
            "xtick.color": "#94A3B8",
            "ytick.color": "#94A3B8",
            "figure.facecolor": chart_bg,
            "axes.facecolor": chart_bg,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
        fig.subplots_adjust(wspace=0.35)

        ax1.plot(
            hist_df["seq"], hist_df["latency_ms"],
            marker="o", markersize=6,
            color="#38BDF8", linewidth=2.2, markerfacecolor="#0284C7",
            markeredgewidth=1.8, markeredgecolor="#E0F2FE",
        )
        ax1.fill_between(hist_df["seq"], hist_df["latency_ms"], alpha=0.18, color="#38BDF8")
        ax1.set_title("End-to-End Latency (ms)", fontsize=11, fontweight="700", pad=12, color=chart_fg)
        ax1.set_xlabel("Packet Sequence", fontsize=9.5, fontweight="600")
        ax1.set_ylabel("Latency (ms)", fontsize=9.5, fontweight="600")

        ax2.bar(hist_df["seq"], hist_df["retransmissions"], color="#F59E0B", width=0.55, edgecolor="#FDE68A", linewidth=1, zorder=3)
        ax2.set_title("TCP Retransmissions per Packet", fontsize=11, fontweight="700", pad=12, color=chart_fg)
        ax2.set_xlabel("Packet Sequence", fontsize=9.5, fontweight="600")
        ax2.set_ylabel("Retransmission Count", fontsize=9.5, fontweight="600")
        ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<p class='section-label'>Raw Telemetry Log</p>", unsafe_allow_html=True)
        st.dataframe(
            hist_df.rename(columns={
                "seq": "Seq", "path": "Path", "delivered": "Delivered",
                "latency_ms": "Latency (ms)", "retransmissions": "Retransmissions",
                "mode": "Routing Mode", "congested": "R2 Congested",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear Telemetry History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown(
            "<div class='info-banner'>No telemetry recorded yet. Run a simulation in the "
            "<strong>Network Topology &amp; ML Routing</strong> tab to populate this view.</div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════
# TAB 4 — SOCKET PROGRAMMING
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<p class='section-label'>Assessment 4 — Host Socket Programming</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-banner'>
        Verifies real Python TCP and UDP socket interoperability over localhost.
        The server is spawned in a background thread, a message is sent, and the server response
        with round-trip time is displayed.
    </div>
    """, unsafe_allow_html=True)

    sock_col1, sock_col2 = st.columns([1, 1], gap="large")

    with sock_col1:
        st.markdown("<div class='panel'><h3>Client Configuration</h3>", unsafe_allow_html=True)
        sock_proto = st.selectbox("Socket Protocol", ["TCP  (port 9091)", "UDP  (port 9092)"])
        sock_msg = st.text_input("Message Payload", "Hello from Computer Networks — Assessment 4")
        sock_send_btn = st.button("Transmit over Local Socket", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'><h3>Protocol Reference</h3>", unsafe_allow_html=True)
        proto_ref = {
            "Property": ["Connection type", "Reliability", "Ordering", "Use case"],
            "TCP": ["Connection-oriented", "Guaranteed", "In-order", "Reliable data transfer"],
            "UDP": ["Connectionless", "Best-effort", "Not guaranteed", "Low-latency streaming"],
        }
        st.dataframe(pd.DataFrame(proto_ref).set_index("Property"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sock_col2:
        st.markdown("<div class='panel'><h3>Server Response</h3>", unsafe_allow_html=True)
        if sock_send_btn:
            server = SocketServer()
            server.start_tcp_server()
            server.start_udp_server()
            time.sleep(0.1)

            client = SocketClient()
            if "TCP" in sock_proto:
                ok, response, rtt = client.send_tcp_message(sock_msg)
            else:
                ok, response, rtt = client.send_udp_message(sock_msg)

            server.stop()

            if ok:
                proto_label = "TCP" if "TCP" in sock_proto else "UDP"
                st.markdown(f"""
                <div class='result-row'>
                    <span class='badge-ok'>ACK</span>
                    <code>{response}</code>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class='kpi-grid' style='margin-top:1rem;'>
                    <div class='kpi-card'>
                        <div class='kpi-label'>Protocol</div>
                        <div class='kpi-value' style='font-size:1.3rem;color:#38BDF8;'>{proto_label}</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-label'>Round-Trip Time</div>
                        <div class='kpi-value' style='font-size:1.3rem;'>{rtt} ms</div>
                    </div>
                    <div class='kpi-card'>
                        <div class='kpi-label'>Status</div>
                        <div class='kpi-value' style='font-size:1.3rem;color:#22C55E;'>OK</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='warn-banner'>Socket error: {response}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:var(--text-muted);font-size:0.9rem;padding:1.5rem 0;text-align:center;font-weight:500;'>"
                "Configure the client and click <strong>Transmit over Local Socket</strong> to test the socket endpoint."
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-size:0.78rem;font-weight:500;color:var(--text-muted);padding-bottom:1rem;'>
    Adaptive Network Routing &amp; Congestion Control &nbsp;|&nbsp;
    Computer Networks Project &nbsp;|&nbsp; VIT University &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)
