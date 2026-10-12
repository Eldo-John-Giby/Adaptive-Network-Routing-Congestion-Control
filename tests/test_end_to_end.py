"""
End-to-End Integration Tests
Verifies the full pipeline:
  - Multi-hop simulated packet transmission
  - Error detection (CRC bit flip) and TCP retransmission recovery
  - Congestion creation on R2, ML detection, and path adaptation
  - Real host socket communication (TCP and UDP)
"""
import time
import pytest
from src.network.topology import NetworkTopology
from src.network.simulator import NetworkSimulator
from src.ml.controller import AdaptiveMLRouter
from src.osi.l4_transport import TransportType
from src.endpoints.socket_server import SocketServer
from src.endpoints.socket_client import SocketClient

def test_simulator_normal_transmission():
    topo = NetworkTopology()
    sim = NetworkSimulator(topo)
    res = sim.transmit_message(
        message="Hello Destination!",
        path=["S", "R1", "R2", "R4", "D"],
        transport_type=TransportType.TCP
    )
    assert res.delivered is True
    assert res.retransmissions == 0
    assert len(res.hops) == 4

def test_crc_error_and_tcp_retransmission():
    topo = NetworkTopology()
    sim = NetworkSimulator(topo)
    # Corrupt first hop CRC
    res = sim.transmit_message(
        message="Message with Error Injection",
        path=["S", "R1", "R2", "R4", "D"],
        transport_type=TransportType.TCP,
        corrupt_first_hop_crc=True,
        max_retransmissions=2
    )
    # With TCP, first attempt fails CRC, second attempt succeeds
    assert res.delivered is True
    assert res.retransmissions >= 1

def test_closed_loop_ml_congestion_adaptation():
    topo = NetworkTopology()
    sim = NetworkSimulator(topo)
    router = AdaptiveMLRouter(topo)

    # 1. Congest Router R2
    topo.stress_router_r2(stress_active=True)

    # 2. Ask ML router to select best path
    chosen_path, eval_res = router.select_best_route("S", "D")

    # 3. Verify R2 was avoided
    assert "R2" not in chosen_path

    # 4. Transmit over the ML-adapted path
    res = sim.transmit_message(
        message="Traffic steered through ML path",
        path=chosen_path,
        transport_type=TransportType.TCP
    )
    assert res.delivered is True
    assert res.selected_path == chosen_path

def test_real_socket_endpoints_tcp_and_udp():
    server = SocketServer(tcp_port=9991, udp_port=9992)
    server.start_tcp_server()
    server.start_udp_server()
    time.sleep(0.15)

    try:
        client = SocketClient()
        # TCP
        tcp_ok, tcp_resp, tcp_rtt = client.send_tcp_message("Test TCP Message", port=9991)
        assert tcp_ok is True
        assert "TCP_ACK" in tcp_resp
        assert tcp_rtt > 0

        # UDP
        udp_ok, udp_resp, udp_rtt = client.send_udp_message("Test UDP Message", port=9992)
        assert udp_ok is True
        assert "UDP_ACK" in udp_resp
        assert udp_rtt > 0
    finally:
        server.stop()
