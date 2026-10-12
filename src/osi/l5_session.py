"""
Layer 5 - Session Layer
Responsible for establishing, managing, synchronizing, and terminating communication sessions.
"""
from dataclasses import dataclass, field
from enum import Enum
import uuid
import time
from typing import Optional, Dict

class SessionState(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    ESTABLISHED = "ESTABLISHED"
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    TERMINATED = "TERMINATED"

@dataclass
class SessionContext:
    session_id: str
    token: str
    state: SessionState
    seq_in_session: int
    started_at: float
    last_active: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "token": self.token,
            "state": self.state.value if isinstance(self.state, SessionState) else str(self.state),
            "seq_in_session": self.seq_in_session,
            "started_at": self.started_at,
            "last_active": self.last_active,
            "metadata": self.metadata
        }

@dataclass
class SessionEnvelope:
    session_id: str
    token: str
    seq_in_session: int
    presentation_payload: str
    session_state: str

class SessionLayer:
    def __init__(self):
        self.active_sessions: Dict[str, SessionContext] = {}

    def create_session(self, metadata: Optional[Dict[str, str]] = None) -> SessionContext:
        session_id = f"SESS-{uuid.uuid4().hex[:8]}"
        token = f"AUTH-{uuid.uuid4().hex[:12]}"
        now = time.time()
        ctx = SessionContext(
            session_id=session_id,
            token=token,
            state=SessionState.ESTABLISHED,
            seq_in_session=0,
            started_at=now,
            last_active=now,
            metadata=metadata or {}
        )
        self.active_sessions[session_id] = ctx
        return ctx

    def wrap_payload(self, ctx: SessionContext, presentation_payload: str) -> SessionEnvelope:
        ctx.seq_in_session += 1
        ctx.last_active = time.time()
        ctx.state = SessionState.ACTIVE
        return SessionEnvelope(
            session_id=ctx.session_id,
            token=ctx.token,
            seq_in_session=ctx.seq_in_session,
            presentation_payload=presentation_payload,
            session_state=ctx.state.value
        )

    def validate_session(self, envelope: SessionEnvelope) -> bool:
        if envelope.session_id not in self.active_sessions:
            # For simulated destination, accept valid session format
            return bool(envelope.session_id and envelope.token)
        ctx = self.active_sessions[envelope.session_id]
        return ctx.token == envelope.token

    def close_session(self, session_id: str):
        if session_id in self.active_sessions:
            self.active_sessions[session_id].state = SessionState.TERMINATED
