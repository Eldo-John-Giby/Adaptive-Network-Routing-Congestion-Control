"""
Layer 7 - Application Layer
Responsible for creating user messages, file payloads, or request/response objects.
"""
from dataclasses import dataclass
import time
import uuid

@dataclass
class ApplicationMessage:
    msg_id: str
    app_type: str        # e.g., "TEXT", "FILE", "TELEMETRY", "COMMAND"
    payload: str         # The raw user message or serialized content
    timestamp: float
    sender_app: str = "Endpoint-A"
    receiver_app: str = "Endpoint-B"

    def to_dict(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "app_type": self.app_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "sender_app": self.sender_app,
            "receiver_app": self.receiver_app
        }

class ApplicationLayer:
    def __init__(self, app_name: str = "AppEndpoint"):
        self.app_name = app_name

    def create_message(self, text: str, app_type: str = "TEXT", receiver: str = "Endpoint-B") -> ApplicationMessage:
        return ApplicationMessage(
            msg_id=f"APP-{uuid.uuid4().hex[:8]}",
            app_type=app_type,
            payload=text,
            timestamp=time.time(),
            sender_app=self.app_name,
            receiver_app=receiver
        )

    def create_file_transfer(self, filename: str, content: str, receiver: str = "Endpoint-B") -> ApplicationMessage:
        payload = f"FILENAME:{filename}\n{content}"
        return self.create_message(text=payload, app_type="FILE", receiver=receiver)
