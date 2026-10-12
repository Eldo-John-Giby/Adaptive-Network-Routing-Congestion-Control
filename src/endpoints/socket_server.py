"""
Socket Server (TCP & UDP)
Demonstrates real host-level socket programming for Assessment 4.
Runs in background threads and handles client connections with echo and ACK responses.
"""
import socket
import threading
import time
from typing import Optional, Callable

class SocketServer:
    def __init__(self, host: str = "127.0.0.1", tcp_port: int = 9091, udp_port: int = 9092):
        self.host = host
        self.tcp_port = tcp_port
        self.udp_port = udp_port

        self._tcp_running = False
        self._udp_running = False
        self._tcp_socket: Optional[socket.socket] = None
        self._udp_socket: Optional[socket.socket] = None

        self._tcp_thread: Optional[threading.Thread] = None
        self._udp_thread: Optional[threading.Thread] = None

        self.messages_received_tcp = []
        self.messages_received_udp = []

    def start_tcp_server(self, on_message: Optional[Callable[[str], None]] = None):
        self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp_socket.bind((self.host, self.tcp_port))
        self._tcp_socket.listen(5)
        self._tcp_running = True

        def loop():
            while self._tcp_running:
                try:
                    self._tcp_socket.settimeout(0.5)
                    conn, addr = self._tcp_socket.accept()
                    data = conn.recv(4096)
                    if data:
                        text = data.decode("utf-8")
                        self.messages_received_tcp.append({"addr": addr, "data": text, "time": time.time()})
                        if on_message:
                            on_message(text)
                        response = f"TCP_ACK: received {len(data)} bytes"
                        conn.sendall(response.encode("utf-8"))
                    conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    break

        self._tcp_thread = threading.Thread(target=loop, daemon=True)
        self._tcp_thread.start()

    def start_udp_server(self, on_message: Optional[Callable[[str], None]] = None):
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.bind((self.host, self.udp_port))
        self._udp_running = True

        def loop():
            while self._udp_running:
                try:
                    self._udp_socket.settimeout(0.5)
                    data, addr = self._udp_socket.recvfrom(4096)
                    if data:
                        text = data.decode("utf-8")
                        self.messages_received_udp.append({"addr": addr, "data": text, "time": time.time()})
                        if on_message:
                            on_message(text)
                        response = f"UDP_ACK: received {len(data)} bytes"
                        self._udp_socket.sendto(response.encode("utf-8"), addr)
                except socket.timeout:
                    continue
                except Exception:
                    break

        self._udp_thread = threading.Thread(target=loop, daemon=True)
        self._udp_thread.start()

    def stop(self):
        self._tcp_running = False
        self._udp_running = False
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass
        if self._udp_socket:
            try:
                self._udp_socket.close()
            except Exception:
                pass
        if self._tcp_thread:
            self._tcp_thread.join(timeout=1.0)
        if self._udp_thread:
            self._udp_thread.join(timeout=1.0)
