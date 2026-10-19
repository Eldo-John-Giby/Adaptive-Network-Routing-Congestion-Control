"""
Network Simulator
Coordinates packet transmission through network paths, tracks end-to-end metrics,
simulates link drops, queue buffers, and provides live telemetry.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import random

from .topology import NetworkTopology
from .node import RouterNode
from ..osi.packet_engine import OSIStack, PacketTrace
from ..osi.l4_transport import TransportType

@dataclass
class HopResult:
    from_node: str
    to_node: str
    success: bool
    delay_ms: float
    queue_occupancy_pct: float
    error_msg: Optional[str] = None

@dataclass
class SimulationResult:
    trace_id: str
    selected_path: List[str]
    delivered: bool
    total_latency_ms: float
    retransmissions: int
    hops: List[HopResult]
    packet_trace: Optional[PacketTrace] = None
    applied_reroute: bool = False
    details: str = ""

class NetworkSimulator:
    def __init__(self, topology: Optional[NetworkTopology] = None):
        self.topology = topology or NetworkTopology()
        self.src_stack = OSIStack(node_id="S", node_ip="10.0.0.1", node_mac="00:00:00:00:00:01")
        self.dst_stack = OSIStack(node_id="D", node_ip="10.0.5.1", node_mac="00:00:00:00:00:06")
        self.total_packets_sent = 0
        self.total_packets_delivered = 0
        self.total_drops = 0
        self.telemetry_history: List[dict] = []

    def transmit_message(self, message: str, path: List[str],
                         transport_type: TransportType = TransportType.TCP,
                         corrupt_first_hop_crc: bool = False,
                         max_retransmissions: int = 3) -> SimulationResult:
        """
        Simulates end-to-end transmission of a message along a given path.
        If CRC error or link drop occurs and transport is TCP, triggers retransmission.
        """
        self.total_packets_sent += 1
        retrans_count = 0
        attempt = 0

        while attempt <= max_retransmissions:
            attempt += 1
            # 1. Encapsulate across all 7 layers at Source
            corrupt = corrupt_first_hop_crc if (attempt == 1 and corrupt_first_hop_crc) else False
            signal, trace = self.src_stack.encapsulate(
                message_text=message,
                dst_ip="10.0.5.1",
                dst_mac="00:00:00:00:00:06",
                transport_type=transport_type,
                corrupt_crc=corrupt
            )

            # 2. Traverse hops
            hop_results = []
            path_success = True
            total_delay = 0.0

            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                link = self.topology.get_link(u, v)
                if not link:
                    path_success = False
                    hop_results.append(HopResult(u, v, False, 0.0, 0.0, f"Link {u}->{v} does not exist"))
                    break

                # Transmit across link
                success, link_delay = link.transmit(packet_size_bytes=signal.bit_count // 8)
                total_delay += link_delay

                # Check router queue
                node_v = self.topology.nodes.get(v)
                queue_pct = 0.0
                if isinstance(node_v, RouterNode):
                    queue_pct = round(node_v.queue_occupancy_ratio * 100, 1)
                    if not node_v.enqueue_packet(signal):
                        # Queue overflow drop!
                        path_success = False
                        self.total_drops += 1
                        hop_results.append(HopResult(u, v, False, link_delay, 100.0, f"Buffer Overflow on {v}"))
                        break
                    node_v.process_and_forward()

                if not success:
                    path_success = False
                    self.total_drops += 1
                    hop_results.append(HopResult(u, v, False, link_delay, queue_pct, f"Link loss on {u}->{v}"))
                    break

                # Check if link causes L2 bit corruption (independent of queue stress)
                if link.should_corrupt():
                    self._corrupt_signal_l2(signal)

                hop_results.append(HopResult(u, v, True, link_delay, queue_pct))

            # 3. Decapsulate at Destination
            if path_success:
                rec_msg, dec_trace = self.dst_stack.decapsulate(signal)
                if dec_trace.decapsulation_success and rec_msg is not None:
                    self.total_packets_delivered += 1
                    sim_res = SimulationResult(
                        trace_id=trace.trace_id,
                        selected_path=path,
                        delivered=True,
                        total_latency_ms=round(total_delay, 2),
                        retransmissions=retrans_count,
                        hops=hop_results,
                        packet_trace=dec_trace,
                        details="Successfully delivered and verified through all 7 layers."
                    )
                    self._record_telemetry(sim_res)
                    return sim_res
                else:
                    # CRC error or parsing error detected at receiver!
                    self.total_drops += 1
                    if transport_type == TransportType.TCP and attempt <= max_retransmissions:
                        retrans_count += 1
                        for hop_idx in range(len(path) - 1):
                            plink = self.topology.get_link(path[hop_idx], path[hop_idx+1])
                            if plink:
                                plink.retransmissions += 1
                        # Backoff wait
                        total_delay += 10.0
                        continue
                    else:
                        sim_res = SimulationResult(
                            trace_id=trace.trace_id,
                            selected_path=path,
                            delivered=False,
                            total_latency_ms=round(total_delay, 2),
                            retransmissions=retrans_count,
                            hops=hop_results,
                            packet_trace=dec_trace,
                            details=f"Packet dropped at receiver: {dec_trace.errors}"
                        )
                        self._record_telemetry(sim_res)
                        return sim_res
            else:
                # Link dropped or queue buffer dropped
                if transport_type == TransportType.TCP and attempt <= max_retransmissions:
                    retrans_count += 1
                    for hop_idx in range(len(path) - 1):
                        plink = self.topology.get_link(path[hop_idx], path[hop_idx+1])
                        if plink:
                            plink.retransmissions += 1
                    total_delay += 15.0
                    continue
                else:
                    sim_res = SimulationResult(
                        trace_id=trace.trace_id,
                        selected_path=path,
                        delivered=False,
                        total_latency_ms=round(total_delay, 2),
                        retransmissions=retrans_count,
                        hops=hop_results,
                        packet_trace=trace,
                        details="Packet lost during path traversal (Max retransmissions exceeded or UDP)."
                    )
                    self._record_telemetry(sim_res)
                    return sim_res

        # Fallback
        return SimulationResult(
            trace_id="TIMEOUT",
            selected_path=path,
            delivered=False,
            total_latency_ms=total_delay,
            retransmissions=retrans_count,
            hops=hop_results,
            details="Transmission failed after maximum retries."
        )

    def _corrupt_signal_l2(self, signal):
        """Injects bit corruption into the L2 serialized frame within a PhysicalSignal."""
        try:
            import json
            raw_str = self.src_stack.l1.bits_to_string(signal.raw_bits)
            l2_dict = json.loads(raw_str)
            l2_dict["is_corrupted"] = True
            if l2_dict.get("payload"):
                orig = l2_dict["payload"]
                corrupted_char = chr(ord(orig[0]) ^ 0x01)
                l2_dict["payload"] = corrupted_char + orig[1:]
            signal.raw_bits = self.src_stack.l1.string_to_bits(json.dumps(l2_dict))
        except Exception:
            pass

    def get_link_telemetry(self, u: str, v: str) -> dict:
        """Returns comprehensive 10-feature telemetry for an individual link."""
        link = self.topology.get_link(u, v)
        if not link:
            return {}
        node_v = self.topology.nodes.get(v)
        q_occ = (node_v.queue_occupancy_ratio * 100.0) if isinstance(node_v, RouterNode) else 0.0
        q_slope = node_v.get_queue_slope() if isinstance(node_v, RouterNode) else 0.0
        rtt_estimate = link.calculate_link_delay() * 2.0
        packet_rate = 150.0 + (q_occ * 20.0)

        return {
            "link_id": f"{u}->{v}",
            "bandwidth_mbps": link.bandwidth_mbps,
            "packet_rate_pps": packet_rate,
            "queue_occupancy_pct": q_occ,
            "rtt_ms": rtt_estimate,
            "loss_rate_pct": link.packet_loss_rate * 100.0,
            "active_connections": link.active_connections,
            "crc_fail_rate": link.crc_fail_rate,
            "retransmission_rate": link.retransmission_rate,
            "rtt_variance": link.rtt_variance,
            "queue_slope": q_slope
        }

    def get_path_telemetry(self, path: List[str]) -> dict:
        """Aggregates cross-layer telemetry features across all links and nodes on a path."""
        total_delay = 0.0
        max_queue_occ = 0.0
        max_loss = 0.0
        max_conn = 1
        min_bw = 1000.0
        bottleneck_node = None
        max_crc_fail_rate = 0.0
        max_retrans_rate = 0.0
        sum_rtt_variance = 0.0
        max_queue_slope = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.topology.get_link(u, v)
            if link:
                total_delay += link.calculate_link_delay()
                max_loss = max(max_loss, link.packet_loss_rate * 100.0)
                max_conn = max(max_conn, link.active_connections)
                min_bw = min(min_bw, link.bandwidth_mbps)
                max_crc_fail_rate = max(max_crc_fail_rate, link.crc_fail_rate)
                max_retrans_rate = max(max_retrans_rate, link.retransmission_rate)
                sum_rtt_variance += link.rtt_variance

            node_v = self.topology.nodes.get(v)
            if isinstance(node_v, RouterNode):
                q_occ = node_v.queue_occupancy_ratio * 100.0
                if q_occ > max_queue_occ:
                    max_queue_occ = q_occ
                    bottleneck_node = v
                total_delay += node_v.queue_length * 2.0
                max_queue_slope = max(max_queue_slope, node_v.get_queue_slope())

        packet_rate = 150.0 + (max_queue_occ * 20.0)

        return {
            "bandwidth_mbps": min_bw,
            "packet_rate_pps": packet_rate,
            "queue_occupancy_pct": max_queue_occ,
            "rtt_ms": total_delay * 2.0,
            "loss_rate_pct": max_loss,
            "active_connections": max_conn,
            "crc_fail_rate": max_crc_fail_rate,
            "retransmission_rate": max_retrans_rate,
            "rtt_variance": round(sum_rtt_variance, 4),
            "queue_slope": max_queue_slope,
            "bottleneck_node": bottleneck_node,
            "accumulated_delay_ms": total_delay
        }

    def _record_telemetry(self, res: SimulationResult):
        record = {
            "timestamp": time.time(),
            "path": " -> ".join(res.selected_path),
            "delivered": res.delivered,
            "latency_ms": res.total_latency_ms,
            "retransmissions": res.retransmissions,
            "r2_queue_pct": self.topology.nodes["R2"].queue_occupancy_ratio * 100 if isinstance(self.topology.nodes["R2"], RouterNode) else 0.0
        }
        self.telemetry_history.append(record)

