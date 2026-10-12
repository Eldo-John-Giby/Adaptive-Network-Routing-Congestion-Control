"""
Socket Client (TCP & UDP)
Demonstrates host socket transmission, measuring latency and receiving transport ACKs.
"""
import socket
import time
from typing import Tuple

class SocketClient:
    def __init__(self, host: str = "127.0.0.1"):
        self.host = host

    def send_tcp_message(self, message: str, port: int = 9091, timeout: float = 2.0) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((self.host, port))
            s.sendall(message.encode("utf-8"))
            response = s.recv(4096).decode("utf-8")
            s.close()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return True, response, round(elapsed_ms, 2)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return False, f"TCP_ERROR: {str(e)}", round(elapsed_ms, 2)

    def send_udp_message(self, message: str, port: int = 9092, timeout: float = 2.0) -> Tuple[bool, str, float]:
        start = time.perf_counter()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(timeout)
            s.sendto(message.encode("utf-8"), (self.host, port))
            response, _ = s.recvfrom(4096)
            s.close()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return True, response.decode("utf-8"), round(elapsed_ms, 2)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return False, f"UDP_ERROR: {str(e)}", round(elapsed_ms, 2)
