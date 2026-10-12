"""
Layer 1 - Physical Layer
Models bit serialization, link bandwidth, transmission delay, and propagation delay.
"""
from dataclasses import dataclass
import time

@dataclass
class PhysicalSignal:
    raw_bits: str              # Binary representation ("0101...")
    bit_count: int
    bandwidth_bps: float       # Channel bandwidth in bits/second
    propagation_delay_ms: float
    transmission_delay_ms: float
    total_latency_ms: float
    timestamp: float

class PhysicalLayer:
    def __init__(self, bandwidth_mbps: float = 10.0, link_distance_km: float = 100.0,
                 propagation_speed_kmps: float = 200000.0):
        self.bandwidth_bps = bandwidth_mbps * 1e6
        self.link_distance_km = link_distance_km
        self.propagation_speed_kmps = propagation_speed_kmps

    def string_to_bits(self, text: str) -> str:
        return ''.join(f"{ord(c):08b}" for c in text)

    def bits_to_string(self, bits: str) -> str:
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)

    def transmit(self, raw_frame_str: str, custom_prop_delay_ms: float = 5.0) -> PhysicalSignal:
        bits = self.string_to_bits(raw_frame_str)
        bit_count = len(bits)
        # T_trans = L / R (in seconds, convert to ms)
        tx_delay_ms = (bit_count / self.bandwidth_bps) * 1000.0
        # T_prop
        prop_delay_ms = custom_prop_delay_ms if custom_prop_delay_ms is not None else (self.link_distance_km / self.propagation_speed_kmps) * 1000.0
        total_latency_ms = tx_delay_ms + prop_delay_ms

        return PhysicalSignal(
            raw_bits=bits,
            bit_count=bit_count,
            bandwidth_bps=self.bandwidth_bps,
            propagation_delay_ms=prop_delay_ms,
            transmission_delay_ms=tx_delay_ms,
            total_latency_ms=total_latency_ms,
            timestamp=time.time()
        )

    def receive(self, signal: PhysicalSignal) -> str:
        return self.bits_to_string(signal.raw_bits)
