"""
Network Nodes: Endpoint Nodes and Router Nodes with Buffer Queues
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import deque
import time

@dataclass
class Node:
    node_id: str
    ip_address: str
    mac_address: str
    is_router: bool = False

    @property
    def ip(self) -> str:
        return self.ip_address

    @property
    def mac(self) -> str:
        return self.mac_address

class RouterNode(Node):
    def __init__(self, node_id: str, ip_address: str, mac_address: str, max_queue_size: int = 50):
        super().__init__(node_id=node_id, ip_address=ip_address, mac_address=mac_address, is_router=True)
        self.max_queue_size = max_queue_size
        self.queue: deque = deque()
        self.packets_forwarded: int = 0
        self.packets_dropped: int = 0
        # Distance-vector routing table: destination_node_id -> {"next_hop": str, "cost": float}
        self.distance_vector_table: Dict[str, dict] = {}

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def queue_occupancy_ratio(self) -> float:
        return len(self.queue) / self.max_queue_size if self.max_queue_size > 0 else 0.0

    def enqueue_packet(self, packet: Any) -> bool:
        """Attempts to buffer packet. Drops packet if queue exceeds max capacity."""
        if len(self.queue) >= self.max_queue_size:
            self.packets_dropped += 1
            return False
        self.queue.append(packet)
        return True

    def process_and_forward(self) -> Optional[Any]:
        if not self.queue:
            return None
        packet = self.queue.popleft()
        self.packets_forwarded += 1
        return packet

    def update_distance_vector(self, dest: str, next_hop: str, cost: float):
        self.distance_vector_table[dest] = {
            "next_hop": next_hop,
            "cost": cost
        }

    def get_status(self) -> dict:
        return {
            "node_id": self.node_id,
            "ip": self.ip_address,
            "queue_len": len(self.queue),
            "max_queue": self.max_queue_size,
            "occupancy_pct": round(self.queue_occupancy_ratio * 100, 1),
            "forwarded": self.packets_forwarded,
            "dropped": self.packets_dropped
        }

class EndpointNode(Node):
    def __init__(self, node_id: str, ip_address: str, mac_address: str):
        super().__init__(node_id=node_id, ip_address=ip_address, mac_address=mac_address, is_router=False)
        self.sent_count = 0
        self.received_count = 0
