"""
Layer 3 - Network Layer
Responsible for logical IP addressing, TTL management, packet fragmentation/reassembly,
and routing path specification.
"""
from dataclasses import dataclass, field
import uuid
import time
from typing import List, Dict, Optional

@dataclass
class IPPacket:
    src_ip: str
    dst_ip: str
    protocol: int        # 6 = TCP, 17 = UDP
    payload: str
    ttl: int = 64
    packet_id: str = field(default_factory=lambda: f"IP-{uuid.uuid4().hex[:6]}")
    route_history: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def decrement_ttl(self, current_node: str) -> bool:
        """Decrements TTL and records hop. Returns False if TTL expired."""
        self.route_history.append(current_node)
        self.ttl -= 1
        return self.ttl > 0

    def to_header_dict(self) -> dict:
        return {
            "packet_id": self.packet_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "protocol": "TCP(6)" if self.protocol == 6 else "UDP(17)" if self.protocol == 17 else str(self.protocol),
            "ttl": self.ttl,
            "hops": len(self.route_history),
            "route_history": self.route_history
        }

class NetworkLayer:
    def __init__(self, node_ip: str = "10.0.0.1"):
        self.node_ip = node_ip
        # Routing table: destination_ip -> {next_hop, cost, path}
        self.routing_table: Dict[str, dict] = {}

    def encapsulate_packet(self, payload: str, dst_ip: str, protocol: int = 6, ttl: int = 64) -> IPPacket:
        pkt = IPPacket(
            src_ip=self.node_ip,
            dst_ip=dst_ip,
            protocol=protocol,
            payload=payload,
            ttl=ttl,
            route_history=[self.node_ip]
        )
        return pkt

    def update_route(self, destination: str, next_hop: str, cost: float, path: Optional[List[str]] = None):
        self.routing_table[destination] = {
            "next_hop": next_hop,
            "cost": cost,
            "path": path or [self.node_ip, next_hop]
        }

    def get_next_hop(self, destination: str) -> Optional[str]:
        if destination in self.routing_table:
            return self.routing_table[destination]["next_hop"]
        return None
