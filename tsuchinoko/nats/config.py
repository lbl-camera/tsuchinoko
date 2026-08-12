"""NATS configuration model."""

from pydantic import BaseModel, Field


class NATSConfig(BaseModel):
    """Configuration for the NATS connection."""
    url: str = ""
    lightfall_prefix: str = "als.7011"
    app_name: str = "tsuchinoko"
    app_version: str = ""
    auth_timeout: float = Field(70.0, description="Auth handshake timeout (>60s for Lightfall trust dialog)")
    connect_timeout: float = Field(5.0, description="Initial connection timeout")
    reconnect: bool = True
