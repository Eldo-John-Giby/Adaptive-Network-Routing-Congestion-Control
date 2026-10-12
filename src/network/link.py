"""
Network Link
Models communication channels between routers/endpoints with bandwidth, base delay,
congestion stress injection, and packet loss.
"""
from dataclasses import dataclass, field
import random
import time

@dataclass
class NetworkLink:
    u: str
    v: str
    base_latency_ms: float
    bandwidth_mbps: float = 10.0
    packet_loss_rate: float = 0.0
    active_connections: int = 1
    # Congestion injection parameters
    stress_factor: float = 1.0       # 1.0 = normal, > 2.0 = congested
    stress_active: bool = False
    packets_transmitted: int = 0
    packets_dropped: int = 0

    @property
    def link_id(self) -> str:
        return f"{self.u}<->{self.v}"

    def set_stress(self, active: bool, stress_multiplier: float = 4.0, additional_loss: float = 0.15):
        """Injects or clears congestion on this link."""
        self.stress_active = active
        if active:
            self.stress_factor = stress_multiplier
            self.packet_loss_rate = min(0.5, self.packet_loss_rate + additional_loss)
            self.active_connections += 8
        else:
            self.stress_factor = 1.0
            self.packet_loss_rate = 0.0
            self.active_connections = max(1, self.active_connections - 8)

    def calculate_link_delay(self, packet_size_bytes: int = 1000) -> float:
        """
        Delay = propagation (base) * stress_factor + transmission delay + random queue jitter.
        """
        bits = packet_size_bytes * 8
        tx_delay_ms = (bits / (self.bandwidth_mbps * 1e6)) * 1000.0
        prop_delay_ms = self.base_latency_ms * self.stress_factor
        jitter_ms = random.uniform(0.1, 1.5) * (self.stress_factor if self.stress_active else 0.5)
        return prop_delay_ms + tx_delay_ms + jitter_ms

    def transmit(self, packet_size_bytes: int = 1000) -> tuple[bool, float]:
        """
        Simulates transmission over the link.
        Returns (success, delay_ms).
        """
        self.packets_transmitted += 1
        # Check packet loss
        if random.random() < self.packet_loss_rate:
            self.packets_dropped += 1
            delay = self.calculate_link_delay(packet_size_bytes)
            return False, delay

        delay = self.calculate_link_delay(packet_size_bytes)
        return True, delay

    def get_metrics(self) -> dict:
        return {
            "link_id": self.link_id,
            "base_latency_ms": self.base_latency_ms,
            "stress_active": self.stress_active,
            "stress_factor": self.stress_factor,
            "loss_rate": round(self.packet_loss_rate, 3),
            "transmitted": self.packets_transmitted,
            "dropped": self.packets_dropped,
            "active_connections": self.active_connections
        }
