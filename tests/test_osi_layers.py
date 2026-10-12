"""
Unit tests for OSI 7-Layer Protocol Stack
"""
import pytest
from src.osi.l7_application import ApplicationLayer
from src.osi.l6_presentation import PresentationLayer
from src.osi.l5_session import SessionLayer, SessionState
from src.osi.l4_transport import TransportLayer, TransportType, TCPFlags
from src.osi.l3_network import NetworkLayer
from src.osi.l2_datalink import DataLinkLayer
from src.osi.l1_physical import PhysicalLayer
from src.osi.packet_engine import OSIStack

def test_l7_application_creation():
    l7 = ApplicationLayer("SourceApp")
    msg = l7.create_message("Hello Network!", app_type="TEXT")
    assert msg.payload == "Hello Network!"
    assert msg.app_type == "TEXT"
    assert msg.msg_id.startswith("APP-")

def test_l6_presentation_compression_and_decompression():
    l7 = ApplicationLayer("TestApp")
    msg = l7.create_message("Repeat" * 100)
    l6 = PresentationLayer(enable_compression=True, compression_threshold=20)
    pdata = l6.encode(msg)
    assert pdata.compressed is True
    decoded_msg = l6.decode(pdata)
    assert decoded_msg.payload == msg.payload

def test_l5_session_lifecycle():
    l5 = SessionLayer()
    ctx = l5.create_session()
    assert ctx.state == SessionState.ESTABLISHED
    assert ctx.token.startswith("AUTH-")

    env = l5.wrap_payload(ctx, '{"test": 123}')
    assert env.seq_in_session == 1
    assert l5.validate_session(env) is True

    l5.close_session(ctx.session_id)
    assert ctx.state == SessionState.TERMINATED

def test_l4_tcp_and_udp():
    l4 = TransportLayer()
    seg = l4.build_segment("PAYLOAD_TCP")
    assert seg.is_valid() is True
    assert seg.flags.ack is True

    udp = l4.build_udp("PAYLOAD_UDP")
    assert udp.is_valid() is True
    assert udp.length == 8 + len("PAYLOAD_UDP")

def test_l3_ip_ttl_decrement():
    l3 = NetworkLayer("10.0.0.1")
    pkt = l3.encapsulate_packet("data", "10.0.5.1", ttl=2)
    assert pkt.decrement_ttl("R1") is True
    assert pkt.ttl == 1
    assert pkt.decrement_ttl("R2") is False  # TTL reaches 0
    assert pkt.ttl == 0

def test_l2_crc_detection_and_corruption():
    l2 = DataLinkLayer()
    frame = l2.frame_packet("IP_PACKET_CONTENT", "00:00:00:00:00:02")
    assert frame.verify_crc() is True

    # Inject bit error
    l2.inject_bit_error(frame)
    assert frame.verify_crc() is False
    valid, err_msg = l2.receive_frame(frame)
    assert valid is False
    assert "CRC_ERROR" in err_msg

def test_l1_physical_latency_calc():
    l1 = PhysicalLayer(bandwidth_mbps=10.0)
    sig = l1.transmit("Hello", custom_prop_delay_ms=5.0)
    assert sig.bit_count == len("Hello") * 8
    assert sig.propagation_delay_ms == 5.0
    assert sig.total_latency_ms > 5.0

def test_full_osistack_end_to_end():
    stack = OSIStack()
    signal, enc_trace = stack.encapsulate("End-to-End Test Payload")
    assert enc_trace.trace_id.startswith("IP-")
    assert enc_trace.l4_trans["proto"] == "TCP"

    msg, dec_trace = stack.decapsulate(signal)
    assert dec_trace.decapsulation_success is True
    assert msg is not None
    assert msg.payload == "End-to-End Test Payload"
