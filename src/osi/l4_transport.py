"""
Layer 4 - Transport Layer
Implements TCP segment handling (Seq/ACK, flags, retransmission, sliding window)
and UDP datagram handling (unreliable, connectionless).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
import time
import json

class TransportType(str, Enum):
    TCP = "TCP"
    UDP = "UDP"

@dataclass
class TCPFlags:
    syn: bool = False
    ack: bool = False
    fin: bool = False
    psh: bool = False
    rst: bool = False

    def to_str(self) -> str:
        flags = []
        if self.syn: flags.append("SYN")
        if self.ack: flags.append("ACK")
        if self.fin: flags.append("FIN")
        if self.psh: flags.append("PSH")
        if self.rst: flags.append("RST")
        return ",".join(flags) if flags else "NONE"

@dataclass
class TCPSegment:
    src_port: int
    dst_port: int
    seq_num: int
    ack_num: int
    flags: TCPFlags
    window_size: int
    payload: str
    checksum: int = 0
    timestamp: float = field(default_factory=time.time)

    def calculate_checksum(self) -> int:
        data = f"{self.src_port}:{self.dst_port}:{self.seq_num}:{self.ack_num}:{self.flags.to_str()}:{self.payload}"
        cksum = 0
        for char in data:
            cksum = (cksum + ord(char)) & 0xFFFF
        return cksum

    def is_valid(self) -> bool:
        return self.checksum == self.calculate_checksum()

@dataclass
class UDPDatagram:
    src_port: int
    dst_port: int
    length: int
    payload: str
    checksum: int = 0
    timestamp: float = field(default_factory=time.time)

    def calculate_checksum(self) -> int:
        data = f"{self.src_port}:{self.dst_port}:{self.length}:{self.payload}"
        cksum = 0
        for char in data:
            cksum = (cksum + ord(char)) & 0xFFFF
        return cksum

    def is_valid(self) -> bool:
        return self.checksum == self.calculate_checksum()

class TransportLayer:
    def __init__(self, transport_type: TransportType = TransportType.TCP, default_port: int = 8080):
        self.transport_type = transport_type
        self.default_port = default_port
        self.current_seq = 1000
        self.current_ack = 0
        self.window_size = 65535
        # Retransmission buffer: seq_num -> (TCPSegment, send_time, retransmission_count)
        self.retransmission_buffer: Dict[int, dict] = {}
        self.rto_seconds: float = 1.0  # Retransmission Timeout

    def build_segment(self, payload: str, src_port: int = 50000, dst_port: int = 8080,
                      flags: Optional[TCPFlags] = None) -> TCPSegment:
        if flags is None:
            flags = TCPFlags(ack=True, psh=True)
        seg = TCPSegment(
            src_port=src_port,
            dst_port=dst_port,
            seq_num=self.current_seq,
            ack_num=self.current_ack,
            flags=flags,
            window_size=self.window_size,
            payload=payload,
            timestamp=time.time()
        )
        seg.checksum = seg.calculate_checksum()
        # buffer for retransmission
        self.retransmission_buffer[self.current_seq] = {
            "segment": seg,
            "send_time": time.time(),
            "attempts": 1
        }
        self.current_seq += max(1, len(payload))
        return seg

    def build_udp(self, payload: str, src_port: int = 50000, dst_port: int = 8080) -> UDPDatagram:
        length = 8 + len(payload.encode("utf-8"))
        dgram = UDPDatagram(
            src_port=src_port,
            dst_port=dst_port,
            length=length,
            payload=payload,
            timestamp=time.time()
        )
        dgram.checksum = dgram.calculate_checksum()
        return dgram

    def acknowledge(self, ack_num: int):
        self.current_ack = ack_num
        # purge acknowledged segments
        to_remove = [seq for seq in self.retransmission_buffer if seq < ack_num]
        for seq in to_remove:
            del self.retransmission_buffer[seq]

    def check_timeouts(self, current_time: Optional[float] = None) -> List[TCPSegment]:
        now = current_time or time.time()
        retransmits = []
        for seq, info in self.retransmission_buffer.items():
            if now - info["send_time"] > self.rto_seconds:
                info["send_time"] = now
                info["attempts"] += 1
                retransmits.append(info["segment"])
        return retransmits
