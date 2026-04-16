"""Centralized configuration management for Tsuchinoko.

This module provides Pydantic-based configuration classes to replace
hardcoded magic numbers throughout the codebase.
"""
from typing import Optional
from pydantic import BaseModel, Field
from tsuchinoko.nats.config import NATSConfig
from tsuchinoko.tiled.config import TiledConfig


class NetworkConfig(BaseModel):
    """Network-related configuration."""
    address: str = 'localhost'
    port: int = Field(5555, ge=1024, le=65535)
    socket_linger: int = Field(5, ge=0)
    recv_timeout_ms: int = Field(5000, ge=100, le=60000)
    queue_timeout_s: float = Field(0.2, ge=0.01, le=10.0)


class UIConfig(BaseModel):
    """UI-related configuration."""
    window_width: int = Field(1700, ge=400)
    window_height: int = Field(1000, ge=300)
    max_log_entries: int = Field(100, ge=10, le=10000)


class CoreConfig(BaseModel):
    """Core experiment loop configuration."""
    sleep_for_fresh_data: float = Field(0.1, ge=0.01, le=10.0)
    checkpoint_interval: int = Field(0, ge=0)


class AdaptiveConfig(BaseModel):
    """Adaptive engine configuration for headless CLI usage."""
    engine_type: str = Field("gpcam", description="Engine type: gpcam or random")
    dimensionality: int = Field(2, ge=1)
    parameter_bounds: list[tuple[float, float]] = Field(
        default=[(0.0, 100.0), (0.0, 100.0)],
        description="Per-axis (min, max) bounds",
    )


class AppConfig(BaseModel):
    """Application-wide configuration."""
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    core: CoreConfig = Field(default_factory=CoreConfig)
    nats: NATSConfig = Field(default_factory=NATSConfig)
    tiled: TiledConfig = Field(default_factory=TiledConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the current application configuration."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global application configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to None (useful for testing)."""
    global _config
    _config = None
