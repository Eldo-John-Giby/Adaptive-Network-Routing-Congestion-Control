"""
Layer 6 - Presentation Layer
Responsible for serialization, character encoding (UTF-8), and optional compression/encryption.
"""
from dataclasses import dataclass
import json
import zlib
import base64
from typing import Tuple
from .l7_application import ApplicationMessage

@dataclass
class PresentationData:
    encoding: str               # "utf-8"
    serialization: str          # "json"
    compressed: bool            # True/False
    original_size: int          # bytes before compression
    encoded_payload: str        # Base64 encoded if compressed, else utf-8 string

class PresentationLayer:
    def __init__(self, enable_compression: bool = False, compression_threshold: int = 128):
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold

    def encode(self, app_msg: ApplicationMessage) -> PresentationData:
        # 1. Serialize Application Message to JSON
        json_str = json.dumps(app_msg.to_dict())
        raw_bytes = json_str.encode("utf-8")
        orig_len = len(raw_bytes)

        # 2. Optional Compression
        is_compressed = False
        if self.enable_compression and orig_len > self.compression_threshold:
            compressed_bytes = zlib.compress(raw_bytes)
            if len(compressed_bytes) < orig_len:
                is_compressed = True
                payload_str = base64.b64encode(compressed_bytes).decode("ascii")
            else:
                payload_str = json_str
        else:
            payload_str = json_str

        return PresentationData(
            encoding="utf-8",
            serialization="json",
            compressed=is_compressed,
            original_size=orig_len,
            encoded_payload=payload_str
        )

    def decode(self, pres_data: PresentationData) -> ApplicationMessage:
        if pres_data.compressed:
            compressed_bytes = base64.b64decode(pres_data.encoded_payload.encode("ascii"))
            raw_bytes = zlib.decompress(compressed_bytes)
            json_str = raw_bytes.decode("utf-8")
        else:
            json_str = pres_data.encoded_payload

        data_dict = json.loads(json_str)
        return ApplicationMessage(**data_dict)
