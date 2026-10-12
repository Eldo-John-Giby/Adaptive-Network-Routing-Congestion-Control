"""
OSI Packet Engine
Coordinates the 7-layer stack, performing full encapsulation (L7 -> L1)
and decapsulation (L1 -> L7), logging every transformation for inspection.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import json

from .l7_application import ApplicationLayer, ApplicationMessage
from .l6_presentation import PresentationLayer, PresentationData
from .l5_session import SessionLayer, SessionContext, SessionEnvelope
from .l4_transport import TransportLayer, TCPSegment, UDPDatagram, TransportType
from .l3_network import NetworkLayer, IPPacket
from .l2_datalink import DataLinkLayer, EthernetFrame
from .l1_physical import PhysicalLayer, PhysicalSignal

@dataclass
class PacketTrace:
    trace_id: str
    l7_app: dict
    l6_pres: dict
    l5_sess: dict
    l4_trans: dict
    l3_net: dict
    l2_link: dict
    l1_phys: dict
    decapsulation_success: bool = True
    errors: list = field(default_factory=list)

class OSIStack:
    def __init__(self, node_id: str = "Endpoint-A", node_ip: str = "10.0.0.1",
                 node_mac: str = "00:00:00:00:00:01", bandwidth_mbps: float = 10.0):
        self.node_id = node_id
        self.node_ip = node_ip
        self.node_mac = node_mac

        self.l7 = ApplicationLayer(app_name=node_id)
        self.l6 = PresentationLayer(enable_compression=True)
        self.l5 = SessionLayer()
        self.l4 = TransportLayer()
        self.l3 = NetworkLayer(node_ip=node_ip)
        self.l2 = DataLinkLayer(node_mac=node_mac)
        self.l1 = PhysicalLayer(bandwidth_mbps=bandwidth_mbps)

        self.active_session: Optional[SessionContext] = None

    def establish_session(self, metadata: Optional[dict] = None) -> SessionContext:
        self.active_session = self.l5.create_session(metadata)
        return self.active_session

    def encapsulate(self, message_text: str, dst_ip: str = "10.0.5.1", dst_mac: str = "00:00:00:00:00:06",
                    transport_type: TransportType = TransportType.TCP,
                    corrupt_crc: bool = False,
                    link_delay_ms: float = 5.0) -> Tuple[PhysicalSignal, PacketTrace]:
        """
        Executes complete encapsulation down through all 7 OSI layers.
        """
        # 1. L7 Application
        l7_msg = self.l7.create_message(text=message_text)

        # 2. L6 Presentation
        l6_data = self.l6.encode(l7_msg)
        l6_serialized = json.dumps({
            "encoding": l6_data.encoding,
            "compressed": l6_data.compressed,
            "payload": l6_data.encoded_payload
        })

        # 3. L5 Session
        if not self.active_session:
            self.establish_session()
        l5_env = self.l5.wrap_payload(self.active_session, l6_serialized)
        l5_serialized = json.dumps({
            "session_id": l5_env.session_id,
            "token": l5_env.token,
            "seq": l5_env.seq_in_session,
            "state": l5_env.session_state,
            "payload": l5_env.presentation_payload
        })

        # 4. L4 Transport
        if transport_type == TransportType.TCP:
            l4_obj = self.l4.build_segment(payload=l5_serialized)
            l4_dict = {
                "proto": "TCP",
                "src_port": l4_obj.src_port,
                "dst_port": l4_obj.dst_port,
                "seq_num": l4_obj.seq_num,
                "ack_num": l4_obj.ack_num,
                "flags": l4_obj.flags.to_str(),
                "checksum": l4_obj.checksum
            }
            l4_serialized = json.dumps({"transport": "TCP", "seq": l4_obj.seq_num, "ack": l4_obj.ack_num, "payload": l4_obj.payload})
        else:
            l4_obj = self.l4.build_udp(payload=l5_serialized)
            l4_dict = {
                "proto": "UDP",
                "src_port": l4_obj.src_port,
                "dst_port": l4_obj.dst_port,
                "length": l4_obj.length,
                "checksum": l4_obj.checksum
            }
            l4_serialized = json.dumps({"transport": "UDP", "len": l4_obj.length, "payload": l4_obj.payload})

        # 5. L3 Network
        l3_proto = 6 if transport_type == TransportType.TCP else 17
        l3_pkt = self.l3.encapsulate_packet(payload=l4_serialized, dst_ip=dst_ip, protocol=l3_proto)
        l3_serialized = json.dumps({
            "packet_id": l3_pkt.packet_id,
            "src_ip": l3_pkt.src_ip,
            "dst_ip": l3_pkt.dst_ip,
            "protocol": l3_pkt.protocol,
            "ttl": l3_pkt.ttl,
            "route_history": l3_pkt.route_history,
            "payload": l3_pkt.payload
        })

        # 6. L2 Data Link
        l2_frame = self.l2.frame_packet(ip_payload=l3_serialized, dst_mac=dst_mac)
        if corrupt_crc:
            self.l2.inject_bit_error(l2_frame)

        l2_serialized = json.dumps({
            "src_mac": l2_frame.src_mac,
            "dst_mac": l2_frame.dst_mac,
            "ethertype": l2_frame.ethertype,
            "crc32": l2_frame.crc32,
            "is_corrupted": l2_frame.is_corrupted,
            "payload": l2_frame.payload
        })

        # 7. L1 Physical
        l1_sig = self.l1.transmit(raw_frame_str=l2_serialized, custom_prop_delay_ms=link_delay_ms)

        trace = PacketTrace(
            trace_id=l3_pkt.packet_id,
            l7_app=l7_msg.to_dict(),
            l6_pres={"encoding": l6_data.encoding, "compressed": l6_data.compressed, "orig_size": l6_data.original_size},
            l5_sess={"session_id": l5_env.session_id, "token": l5_env.token, "seq": l5_env.seq_in_session},
            l4_trans=l4_dict,
            l3_net=l3_pkt.to_header_dict(),
            l2_link=l2_frame.to_dict(),
            l1_phys={
                "bit_count": l1_sig.bit_count,
                "tx_delay_ms": round(l1_sig.transmission_delay_ms, 4),
                "prop_delay_ms": round(l1_sig.propagation_delay_ms, 4),
                "total_latency_ms": round(l1_sig.total_latency_ms, 4),
                "sample_bits": l1_sig.raw_bits[:48] + "..."
            }
        )

        return l1_sig, trace

    def decapsulate(self, signal: PhysicalSignal) -> Tuple[Optional[ApplicationMessage], PacketTrace]:
        """
        Executes complete decapsulation upwards from L1 to L7.
        """
        errors = []
        # L1 Physical
        raw_frame_str = self.l1.receive(signal)

        # L2 Data Link
        try:
            l2_dict = json.loads(raw_frame_str)
            frame = EthernetFrame(
                src_mac=l2_dict["src_mac"],
                dst_mac=l2_dict["dst_mac"],
                ethertype=l2_dict["ethertype"],
                payload=l2_dict["payload"],
                crc32=l2_dict["crc32"],
                is_corrupted=l2_dict.get("is_corrupted", False)
            )
            valid, l3_raw = self.l2.receive_frame(frame)
            if not valid:
                errors.append(l3_raw)
                return None, PacketTrace(
                    trace_id="CORRUPTED",
                    l7_app={}, l6_pres={}, l5_sess={}, l4_trans={}, l3_net={},
                    l2_link=frame.to_dict(),
                    l1_phys={"bit_count": signal.bit_count, "latency_ms": signal.total_latency_ms},
                    decapsulation_success=False,
                    errors=errors
                )
        except Exception as e:
            errors.append(f"L2 parsing failed: {e}")
            return None, PacketTrace(trace_id="ERR-L2", l7_app={}, l6_pres={}, l5_sess={}, l4_trans={}, l3_net={}, l2_link={}, l1_phys={}, decapsulation_success=False, errors=errors)

        # L3 Network
        try:
            l3_dict = json.loads(l3_raw)
            pkt = IPPacket(
                src_ip=l3_dict["src_ip"],
                dst_ip=l3_dict["dst_ip"],
                protocol=l3_dict["protocol"],
                payload=l3_dict["payload"],
                ttl=l3_dict["ttl"],
                packet_id=l3_dict["packet_id"],
                route_history=l3_dict.get("route_history", [])
            )
            l4_raw = pkt.payload
        except Exception as e:
            errors.append(f"L3 parsing failed: {e}")
            return None, PacketTrace(trace_id="ERR-L3", l7_app={}, l6_pres={}, l5_sess={}, l4_trans={}, l3_net={}, l2_link=frame.to_dict(), l1_phys={}, decapsulation_success=False, errors=errors)

        # L4 Transport
        try:
            l4_dict = json.loads(l4_raw)
            l5_raw = l4_dict["payload"]
        except Exception as e:
            errors.append(f"L4 parsing failed: {e}")
            return None, PacketTrace(trace_id=pkt.packet_id, l7_app={}, l6_pres={}, l5_sess={}, l4_trans={}, l3_net=pkt.to_header_dict(), l2_link=frame.to_dict(), l1_phys={}, decapsulation_success=False, errors=errors)

        # L5 Session
        try:
            l5_dict = json.loads(l5_raw)
            l6_raw = l5_dict["payload"]
        except Exception as e:
            errors.append(f"L5 parsing failed: {e}")
            return None, PacketTrace(trace_id=pkt.packet_id, l7_app={}, l6_pres={}, l5_sess={}, l4_trans=l4_dict, l3_net=pkt.to_header_dict(), l2_link=frame.to_dict(), l1_phys={}, decapsulation_success=False, errors=errors)

        # L6 Presentation
        try:
            l6_dict = json.loads(l6_raw)
            pdata = PresentationData(
                encoding=l6_dict["encoding"],
                serialization="json",
                compressed=l6_dict["compressed"],
                original_size=0,
                encoded_payload=l6_dict["payload"]
            )
            # L7 Application
            app_msg = self.l6.decode(pdata)
        except Exception as e:
            errors.append(f"L6/L7 decoding failed: {e}")
            return None, PacketTrace(trace_id=pkt.packet_id, l7_app={}, l6_pres={}, l5_sess=l5_dict, l4_trans=l4_dict, l3_net=pkt.to_header_dict(), l2_link=frame.to_dict(), l1_phys={}, decapsulation_success=False, errors=errors)

        trace = PacketTrace(
            trace_id=pkt.packet_id,
            l7_app=app_msg.to_dict(),
            l6_pres=l6_dict,
            l5_sess={"session_id": l5_dict.get("session_id"), "seq": l5_dict.get("seq")},
            l4_trans=l4_dict,
            l3_net=pkt.to_header_dict(),
            l2_link=frame.to_dict(),
            l1_phys={"bit_count": signal.bit_count, "latency_ms": signal.total_latency_ms},
            decapsulation_success=True,
            errors=[]
        )
        return app_msg, trace
