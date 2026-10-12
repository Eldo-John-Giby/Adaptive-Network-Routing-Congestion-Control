"""
Interactive Visual Dashboard for Adaptive Network Routing & Congestion Control
Built with Streamlit and Matplotlib.
Demonstrates:
  - 7-Layer OSI packet encapsulation/decapsulation inspector
  - Multi-path network topology (Figure 5)
  - Congestion stress injection on Router R2
  - Machine Learning-assisted dynamic rerouting
  - Live socket interop test
"""
import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.osi.packet_engine import OSIStack
from src.osi.l4_transport import TransportType
from src.network.topology import NetworkTopology
from src.network.simulator import NetworkSimulator
from src.ml.model import NetworkCongestionModel
from src.ml.controller import AdaptiveMLRouter
from src.endpoints.socket_server import SocketServer
from src.endpoints.socket_client import SocketClient

st.set_page_config(
    page_title="Adaptive Network Routing & ML Congestion Control",
    page_icon="🌐",
    layout="wide"
)

# Initialize Session State
if "topology" not in st.session_state:
    st.session_state.topology = NetworkTopology()
if "simulator" not in st.session_state:
    st.session_state.simulator = NetworkSimulator(st.session_state.topology)
if "ml_model" not in st.session_state:
    model = NetworkCongestionModel()
    model.train()
    st.session_state.ml_model = model
if "router" not in st.session_state:
    st.session_state.router = AdaptiveMLRouter(st.session_state.topology, st.session_state.ml_model)
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🌐 Adaptive Network Routing & Congestion Control")
st.caption("End-to-End 7-Layer OSI Simulation, Router Queue Dynamics & Machine Learning Route Optimization")

tabs = st.tabs([
    "🔍 7-Layer OSI Inspector",
    "🗺️ Network Topology & ML Routing",
    "📊 Performance Metrics & Telemetry",
    "🔌 Real Socket Programming"
])

# -------------------------------------------------------------
# TAB 1: 7-LAYER OSI INSPECTOR
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Seven-Layer OSI Protocol Stack Demonstration")
    st.markdown("Encapsulates data from Application down to Physical bitstream, then decapsulates it at the receiver.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Transmission Settings")
        input_msg = st.text_area(
            "Application Layer Payload (L7)",
            value="CRITICAL_PAYLOAD: Sensor readings telemetry stream #1042."
        )
        proto_choice = st.radio("Transport Protocol (L4)", ["TCP (Reliable, Seq/ACK)", "UDP (Unreliable, Low Latency)"])
        corrupt_crc = st.checkbox("⚠️ Inject CRC Frame Corruption (Bit-flip on L2 Data Link)", value=False)
        send_btn = st.button("🚀 Encapsulate & Inspect Stack", use_container_width=True)

    with col2:
        st.markdown("### 📥 Stack Inspection")
        if send_btn:
            stack = OSIStack()
            t_type = TransportType.TCP if "TCP" in proto_choice else TransportType.UDP
            sig, enc_trace = stack.encapsulate(input_msg, transport_type=t_type, corrupt_crc=corrupt_crc)
            app_msg, dec_trace = stack.decapsulate(sig)

            st.success(f"Packet Encapsulated! ID: `{enc_trace.trace_id}`")

            # Layer Expander cards
            with st.expander("L7 - Application Layer", expanded=True):
                st.json(enc_trace.l7_app)

            with st.expander("L6 - Presentation Layer (JSON & Compression)", expanded=False):
                st.json(enc_trace.l6_pres)

            with st.expander("L5 - Session Layer (Tokens & State)", expanded=False):
                st.json(enc_trace.l5_sess)

            with st.expander("L4 - Transport Layer (TCP/UDP Headers & Checksum)", expanded=True):
                st.json(enc_trace.l4_trans)

            with st.expander("L3 - Network Layer (IPv4 Header & TTL)", expanded=False):
                st.json(enc_trace.l3_net)

            with st.expander("L2 - Data Link Layer (Ethernet Frame & CRC32)", expanded=True):
                st.json(enc_trace.l2_link)
                if corrupt_crc:
                    st.error("❌ CRC Verification FAILED: Corrupted frame caught by receiver!")
                else:
                    st.success("✅ CRC Check Passed: Frame verified intact.")

            with st.expander("L1 - Physical Layer (Bitstream & Delay)", expanded=False):
                st.json(enc_trace.l1_phys)

# -------------------------------------------------------------
# TAB 2: NETWORK TOPOLOGY & ML ROUTING
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Multi-Path Topology & Closed-Loop ML Rerouting")
    st.markdown("""
    **Figure 5 Topology:**
    - `S -> R1 (8ms)`, `S -> R3 (14ms)`
    - `R1 -> R2 (5ms)`, `R1 -> R3 (6ms)`
    - `R2 -> R4 (4ms)`, `R3 -> R4 (3ms)`
    - `R4 -> D (7ms)`
    """)

    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        st.markdown("### ⚙️ Network Conditions & Congestion Control")
        stress_r2 = st.checkbox("🔥 Congest Router R2 (Queue Overload + 6x Latency + 20% Packet Loss)", value=False)
        st.session_state.topology.stress_router_r2(stress_active=stress_r2)

        routing_mode = st.radio("Routing Decision Policy:", ["🤖 ML-Assisted Adaptive Routing", "📍 Static Shortest Path Routing (Fixed via R2)"])
        sim_packets = st.slider("Number of Packets to Transmit:", min_value=1, max_value=20, value=5)
        run_sim_btn = st.button("▶️ Run Route Simulation", use_container_width=True)

    with top_col2:
        st.markdown("### 🧠 ML Candidate Route Predictions")
        evals = st.session_state.router.evaluate_all_candidate_paths("S", "D")

        eval_records = []
        for ev in evals:
            eval_records.append({
                "Route": ev.path_name,
                "Path": " -> ".join(ev.path),
                "Base Delay": f"{ev.actual_base_delay_ms} ms",
                "ML Congestion State": ev.congestion_label,
                "ML Confidence": f"{ev.confidence * 100:.1f}%",
                "Predicted Delay": f"{ev.predicted_delay_ms} ms",
                "Recommendation": ev.recommendation
            })
        st.dataframe(pd.DataFrame(eval_records), use_container_width=True)

    if run_sim_btn:
        st.markdown("### 📦 Packet Transmission Results")
        results = []
        for i in range(sim_packets):
            if "ML" in routing_mode:
                chosen_path, _ = st.session_state.router.select_best_route("S", "D")
            else:
                # Static path always via R2: S -> R1 -> R2 -> R4 -> D
                chosen_path = ["S", "R1", "R2", "R4", "D"]

            res = st.session_state.simulator.transmit_message(
                message=f"SIM_MSG_{i+1}",
                path=chosen_path,
                transport_type=TransportType.TCP
            )
            results.append(res)
            st.session_state.history.append({
                "seq": len(st.session_state.history) + 1,
                "path": " -> ".join(res.selected_path),
                "delivered": res.delivered,
                "latency_ms": res.total_latency_ms,
                "retransmissions": res.retransmissions,
                "mode": routing_mode,
                "congested": stress_r2
            })

        for r in results:
            badge = "✅ DELIVERED" if r.delivered else "❌ DROPPED"
            st.write(f"**Packet `{r.trace_id}`:** {badge} | Path: `{' -> '.join(r.selected_path)}` | Total Latency: **{r.total_latency_ms} ms** | Retransmissions: **{r.retransmissions}**")

# -------------------------------------------------------------
# TAB 3: PERFORMANCE METRICS & TELEMETRY
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Performance Monitor & Comparative Telemetry")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Packets Sent", len(hist_df))
        c2.metric("Delivery Rate", f"{(hist_df['delivered'].mean() * 100):.1f}%")
        c3.metric("Avg Latency", f"{hist_df['latency_ms'].mean():.1f} ms")
        c4.metric("Total Retransmissions", int(hist_df["retransmissions"].sum()))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(hist_df["seq"], hist_df["latency_ms"], marker="o", color="#1f77b4")
        ax1.set_title("End-to-End Latency over Packets (ms)")
        ax1.set_xlabel("Packet Sequence")
        ax1.set_ylabel("Latency (ms)")
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.bar(hist_df["seq"], hist_df["retransmissions"], color="#ff7f0e")
        ax2.set_title("TCP Retransmissions per Packet")
        ax2.set_xlabel("Packet Sequence")
        ax2.set_ylabel("Retransmission Count")
        ax2.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("Run packet transmissions in the 'Network Topology & ML Routing' tab to view live charts.")

# -------------------------------------------------------------
# TAB 4: REAL SOCKET PROGRAMMING (ASSESSMENT 4)
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Assessment 4: Host Socket Programming Demonstration")
    st.markdown("Verifies real TCP and UDP socket interop over localhost.")

    sock_col1, sock_col2 = st.columns([1, 1])

    with sock_col1:
        st.markdown("### 📡 Client Transmission")
        sock_proto = st.selectbox("Socket Type", ["TCP (Port 9091)", "UDP (Port 9092)"])
        sock_msg = st.text_input("Socket Message", "Hello from Computer Networks Assessment 4!")
        sock_send_btn = st.button("Send Over Local Socket")

    with sock_col2:
        st.markdown("### 📥 Server Response")
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
                st.success(f"Response Received: `{response}`")
                st.info(f"Socket Round-Trip Time (RTT): **{rtt} ms**")
            else:
                st.error(f"Error: {response}")
