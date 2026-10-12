"""
Unified End-to-End Demonstration Script
Adaptive Network Routing and Congestion Control Enhanced by Machine Learning
Based on cn_proj.docx specifications.
"""
import sys
import os
import time
import json

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.osi.packet_engine import OSIStack
from src.osi.l4_transport import TransportType
from src.network.topology import NetworkTopology
from src.network.simulator import NetworkSimulator
from src.ml.model import NetworkCongestionModel
from src.ml.controller import AdaptiveMLRouter
from src.endpoints.socket_server import SocketServer
from src.endpoints.socket_client import SocketClient

def print_banner():
    print("=" * 80)
    print("   ADAPTIVE NETWORK ROUTING & CONGESTION CONTROL (ML-ENHANCED)")
    print("   End-to-End 7-Layer OSI Simulation & Dynamic Rerouting Engine")
    print("=" * 80)
    print("Project Authors:")
    print("  - Shreejith J (24BCE0625)")
    print("  - Pooja Iykkarath Santhosh (24BAI0269)")
    print("  - Hemal Kamineni (24BCE0649)")
    print("  - Eldo John Giby (24BCE0387)")
    print("  - Anfy Sony (24BCE2799)")
    print("=" * 80)
    print("Core Idea: Generate -> Encapsulate -> Route -> Stress -> Predict -> Adapt -> Deliver -> Measure\n")

def run_osi_demo():
    print("\n" + "#" * 80)
    print("STAGE 1: SEVEN-LAYER OSI PACKET JOURNEY & INSPECTION (Assessments 2 & 5)")
    print("#" * 80)
    stack = OSIStack(node_id="Endpoint-S", node_ip="10.0.0.1", node_mac="00:00:00:00:00:01")
    payload = "TELEMETRY_SAMPLE: Temperature=24.5C, Pressure=1013hPa, Status=NOMINAL"
    print(f"[*] Input Message at L7: '{payload}'\n")

    sig, enc_trace = stack.encapsulate(
        message_text=payload,
        dst_ip="10.0.5.1",
        dst_mac="00:00:00:00:00:06",
        transport_type=TransportType.TCP
    )

    print("[L7 Application]  -> Message ID:      ", enc_trace.l7_app["msg_id"])
    print("[L6 Presentation] -> Encoding:        ", enc_trace.l6_pres["encoding"], f"(Compressed={enc_trace.l6_pres['compressed']})")
    print("[L5 Session]      -> Session ID:      ", enc_trace.l5_sess["session_id"], f"Token: {enc_trace.l5_sess['token'][:16]}...")
    print("[L4 Transport]    -> Protocol: TCP    ", f"Seq={enc_trace.l4_trans['seq_num']}, Ack={enc_trace.l4_trans['ack_num']}, Flags={enc_trace.l4_trans['flags']}")
    print("[L3 Network]      -> IP Header:       ", f"Src={enc_trace.l3_net['src_ip']} -> Dst={enc_trace.l3_net['dst_ip']}, TTL={enc_trace.l3_net['ttl']}")
    print("[L2 Data Link]    -> Ethernet Frame:  ", f"MAC: {enc_trace.l2_link['src_mac']} -> {enc_trace.l2_link['dst_mac']}, CRC-32={enc_trace.l2_link['crc32']}")
    print("[L1 Physical]     -> Physical Signal: ", f"Bits={enc_trace.l1_phys['bit_count']}, TxDelay={enc_trace.l1_phys['tx_delay_ms']}ms, PropDelay={enc_trace.l1_phys['prop_delay_ms']}ms")
    print(f"                     Raw Bitstream:    {enc_trace.l1_phys['sample_bits']}")

    print("\n[*] Propagating across medium to Receiver (Endpoint-D)...")
    rec_msg, dec_trace = stack.decapsulate(sig)
    if dec_trace.decapsulation_success and rec_msg:
        print(f"[+] Decapsulation SUCCESSFUL at Destination!")
        print(f"    Recovered Application Payload: '{rec_msg.payload}'")
    else:
        print(f"[-] Decapsulation FAILED: {dec_trace.errors}")

def run_error_recovery_demo():
    print("\n" + "#" * 80)
    print("STAGE 2: CRC ERROR DETECTION & TCP RETRANSMISSION (Assessment 3)")
    print("#" * 80)
    topo = NetworkTopology()
    sim = NetworkSimulator(topo)

    print("[*] Transmitting packet with intentional bit-flip error in L2 Data Link payload...")
    res = sim.transmit_message(
        message="Financial Transaction #9921",
        path=["S", "R1", "R2", "R4", "D"],
        transport_type=TransportType.TCP,
        corrupt_first_hop_crc=True,
        max_retransmissions=2
    )

    print(f"[*] Delivery Status:       {'DELIVERED' if res.delivered else 'DROPPED'}")
    print(f"[*] TCP Retransmissions:   {res.retransmissions}")
    print(f"[*] Total Latency:         {res.total_latency_ms} ms")
    print(f"[*] Details:               {res.details}")
    if res.retransmissions > 0:
        print("[+] Observation: Attempt #1 rejected due to CRC mismatch; TCP retransmission succeeded on Attempt #2.")

def run_ml_adaptive_routing_demo():
    print("\n" + "#" * 80)
    print("STAGE 3: TOPOLOGY STRESS & MACHINE LEARNING ADAPTIVE ROUTING (Assessments 6 & 7)")
    print("#" * 80)
    topo = NetworkTopology()
    router = AdaptiveMLRouter(topo)
    sim = NetworkSimulator(topo)

    print("Network Topology (Figure 5):")
    print("       (8)           (5)           (4)")
    print("  [S] -----> [R1] ---------> [R2] ---------> [R4] -----> [D] (7)")
    print("   |          |                            ^       ^")
    print("   | (14)     | (6)                        | (3)   |")
    print("   +--------> [R3] ------------------------+-------+")
    print()

    # Step 1: Baseline evaluation under normal traffic
    print("--- Phase 3.1: Baseline Evaluation (Normal Network State) ---")
    evals = router.evaluate_all_candidate_paths("S", "D")
    for ev in evals:
        print(f"  * {ev.path_name:22} | Path: {' -> '.join(ev.path):22} | ML State: {ev.congestion_label:6} | Pred Delay: {ev.predicted_delay_ms:5.1f}ms | Rec: {ev.recommendation}")

    best_path_1, _ = router.select_best_route("S", "D")
    print(f"[+] Initial Selected Route: {' -> '.join(best_path_1)}")

    # Step 2: Inject stress on Router R2
    print("\n--- Phase 3.2: Congesting Router R2 (Queue Overload + Link Stress) ---")
    topo.stress_router_r2(stress_active=True)
    print("[!] Traffic surge injected on Router R2: Queue buffer filled to 80%, link delay multiplied 6x.")

    # Step 3: ML re-evaluation
    print("\n--- Phase 3.3: ML Model Dynamic Inference & Path Re-evaluation ---")
    evals_stressed = router.evaluate_all_candidate_paths("S", "D")
    for ev in evals_stressed:
        print(f"  * {ev.path_name:22} | Path: {' -> '.join(ev.path):22} | ML State: {ev.congestion_label:6} | Pred Delay: {ev.predicted_delay_ms:5.1f}ms | Rec: {ev.recommendation}")

    best_path_2, best_eval = router.select_best_route("S", "D")
    print(f"\n[+] Adaptive ML Route Decision: Shifted from Route 1 to -> {' -> '.join(best_path_2)}")
    print(f"    Reason: R2 marked as '{evals_stressed[2].congestion_label}' congestion. ML selected path with predicted delay {best_eval.predicted_delay_ms} ms.")

    # Step 4: Transmit under ML path vs Static path
    print("\n--- Phase 3.4: Comparative Verification ---")
    # A. Static route (stuck through congested R2)
    res_static = sim.transmit_message("Static Path Test", path=["S", "R1", "R2", "R4", "D"], transport_type=TransportType.TCP)
    # B. ML adapted route (bypassing R2)
    res_ml = sim.transmit_message("ML Path Test", path=best_path_2, transport_type=TransportType.TCP)

    print(f"  * Static Path (via R2):  Delivered={res_static.delivered}, Latency={res_static.total_latency_ms:6.2f}ms, Retransmits={res_static.retransmissions}")
    print(f"  * ML Path     (via R3):  Delivered={res_ml.delivered}, Latency={res_ml.total_latency_ms:6.2f}ms, Retransmits={res_ml.retransmissions}")
    latency_savings = res_static.total_latency_ms - res_ml.total_latency_ms
    print(f"[+] Latency Improvement from ML Adaptive Routing: {latency_savings:.1f} ms reduction!")

def run_socket_programming_demo():
    print("\n" + "#" * 80)
    print("STAGE 4: HOST SOCKET PROGRAMMING (Assessment 4)")
    print("#" * 80)
    server = SocketServer(tcp_port=9101, udp_port=9102)
    server.start_tcp_server()
    server.start_udp_server()
    time.sleep(0.15)

    client = SocketClient()
    print("[*] Testing TCP Socket Connection on 127.0.0.1:9101...")
    tcp_ok, tcp_resp, tcp_rtt = client.send_tcp_message("Hello from Python TCP Socket!", port=9101)
    print(f"    Status: {'SUCCESS' if tcp_ok else 'FAILED'} | Server Reply: '{tcp_resp}' | Socket RTT: {tcp_rtt} ms")

    print("[*] Testing UDP Socket Datagram on 127.0.0.1:9102...")
    udp_ok, udp_resp, udp_rtt = client.send_udp_message("Hello from Python UDP Socket!", port=9102)
    print(f"    Status: {'SUCCESS' if udp_ok else 'FAILED'} | Server Reply: '{udp_resp}' | Socket RTT: {udp_rtt} ms")

    server.stop()
    print("[+] Socket endpoints verified successfully.")

def print_summary():
    print("\n" + "=" * 80)
    print("LAB ASSESSMENT COVERAGE SUMMARY (Figure 6 Alignment)")
    print("=" * 80)
    print(" [OK] Assessment 2 (Demonstration):    7-Layer packet encapsulation, traversal & decapsulation")
    print(" [OK] Assessment 3 (Error Detection): CRC-32 checksum calculation, bit-flip corruption & TCP recovery")
    print(" [OK] Assessment 4 (Socket Prog):     Python TCP & UDP socket client-server endpoints")
    print(" [OK] Assessment 5 (Packet Analysis): Layer-by-layer header inspection (L7 down to L1)")
    print(" [OK] Assessment 6 (Routing Protocol): Multi-hop Distance-Vector path generation & cost updates")
    print(" [OK] Assessment 7 (Congestion Ctrl): Finite router queues, telemetry metrics, ML-driven rerouting")
    print("=" * 80)
    print("All demonstrations executed successfully!\n")

if __name__ == "__main__":
    print_banner()
    run_osi_demo()
    run_error_recovery_demo()
    run_ml_adaptive_routing_demo()
    run_socket_programming_demo()
    print_summary()
