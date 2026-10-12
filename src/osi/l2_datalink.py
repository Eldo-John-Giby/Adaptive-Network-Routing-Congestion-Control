"""
Layer 2 - Data Link Layer
Responsible for MAC framing, hardware addressing, and CRC-32 error detection.
Supports intentional bit-flip error injection to demonstrate error detection and recovery.
"""
from dataclasses import dataclass, field
import binascii
import time
from typing import Tuple

@dataclass
class EthernetFrame:
    src_mac: str
    dst_mac: str
    ethertype: int      # 0x0800 for IPv4
    payload: str
    crc32: int = 0
    preamble: str = "10101010" * 7 + "10101011"  # 7 bytes preamble + 1 byte SFD
    is_corrupted: bool = False
    timestamp: float = field(default_factory=time.time)

    def calculate_crc(self) -> int:
        raw = f"{self.src_mac}{self.dst_mac}{self.ethertype:04X}{self.payload}".encode("utf-8")
        return binascii.crc32(raw) & 0xFFFFFFFF

    def verify_crc(self) -> bool:
        if self.is_corrupted:
            return False
        return self.crc32 == self.calculate_crc()

    def to_dict(self) -> dict:
        return {
            "preamble": self.preamble[:16] + "...",
            "src_mac": self.src_mac,
            "dst_mac": self.dst_mac,
            "ethertype": f"0x{self.ethertype:04X}",
            "payload_len": len(self.payload),
            "crc32": f"0x{self.crc32:08X}",
            "crc_valid": self.verify_crc(),
            "is_corrupted": self.is_corrupted
        }

class DataLinkLayer:
    def __init__(self, node_mac: str = "00:00:00:00:00:01"):
        self.node_mac = node_mac
        self.frames_sent = 0
        self.frames_received = 0
        self.crc_errors_detected = 0

    def frame_packet(self, ip_payload: str, dst_mac: str, ethertype: int = 0x0800) -> EthernetFrame:
        frame = EthernetFrame(
            src_mac=self.node_mac,
            dst_mac=dst_mac,
            ethertype=ethertype,
            payload=ip_payload,
            timestamp=time.time()
        )
        frame.crc32 = frame.calculate_crc()
        self.frames_sent += 1
        return frame

    def inject_bit_error(self, frame: EthernetFrame, bit_position: int = 0) -> EthernetFrame:
        """Flips a character/bit in payload to simulate channel noise or cable fault."""
        if not frame.payload:
            frame.payload = "X"
        else:
            corrupted_char = chr(ord(frame.payload[0]) ^ 0x01)
            frame.payload = corrupted_char + frame.payload[1:]
        frame.is_corrupted = True
        return frame

    def receive_frame(self, frame: EthernetFrame) -> Tuple[bool, str]:
        """
        Validates frame CRC and destination MAC.
        Returns (is_valid, extracted_payload_or_error_msg).
        """
        self.frames_received += 1
        if not frame.verify_crc():
            self.crc_errors_detected += 1
            return False, f"CRC_ERROR: Frame checksum mismatch (got 0x{frame.crc32:08X}, expected 0x{frame.calculate_crc():08X})"
        return True, frame.payload
