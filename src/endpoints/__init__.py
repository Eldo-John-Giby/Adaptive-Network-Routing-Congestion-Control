"""
Socket Programming Endpoints (TCP & UDP)
Demonstrating real socket communication corresponding to Assessment 4.
"""
from .socket_server import SocketServer
from .socket_client import SocketClient

__all__ = ["SocketServer", "SocketClient"]
