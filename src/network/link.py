"""
Network Link
Models communication channels between routers/endpoints with bandwidth, base delay,
congestion stress injection, and packet loss.
"""
from dataclasses import dataclass, field
from collections import deque
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
    # Corruption injection parameters
    bit_error_rate: float = 0.0      # independent of queue stress
    crc_errors: int = 0
    retransmissions: int = 0
    recent_delays: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def link_id(self) -> str:
        return f"{self.u}<->{self.v}"

    def inject_corruption(self, bit_error_rate: float):
        """Injects bit error rate corruption on this link, independent of queue stress."""
        self.bit_error_rate = max(0.0, float(bit_error_rate))

    def should_corrupt(self) -> bool:
        """Determines if a packet across this link suffers bit corruption."""
        if self.bit_error_rate > 0.0 and random.random() < self.bit_error_rate:
            self.crc_errors += 1
            return True
        return False

    @property
    def crc_fail_rate(self) -> float:
        return round(self.crc_errors / max(1, self.packets_transmitted), 4)

    @property
    def retransmission_rate(self) -> float:
        return round(self.retransmissions / max(1, self.packets_transmitted), 4)

    @property
    def rtt_variance(self) -> float:
        if len(self.recent_delays) >= 2:
            vals = list(self.recent_delays)
            mean = sum(vals) / len(vals)
            return round(sum((x - mean) ** 2 for x in vals) / len(vals), 4)
        return 0.0

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
        delay = self.calculate_link_delay(packet_size_bytes)
        self.recent_delays.append(delay)

        # Check packet loss due to congestion/buffer loss
        if random.random() < self.packet_loss_rate:
            self.packets_dropped += 1
            return False, delay

        return True, delay

    def get_metrics(self) -> dict:
        return {
            "link_id": self.link_id,
            "base_latency_ms": self.base_latency_ms,
            "stress_active": self.stress_active,
            "stress_factor": self.stress_factor,
            "loss_rate": round(self.packet_loss_rate, 3),
            "bit_error_rate": round(self.bit_error_rate, 3),
            "crc_errors": self.crc_errors,
            "crc_fail_rate": self.crc_fail_rate,
            "retransmission_rate": self.retransmission_rate,
            "rtt_variance": self.rtt_variance,
            "transmitted": self.packets_transmitted,
            "dropped": self.packets_dropped,
            "active_connections": self.active_connections
        }

def inject_corruption(link: NetworkLink, bit_error_rate: float):
    """Injects bit error rate corruption on a network link."""
    link.inject_corruption(bit_error_rate)

