"""Tiled configuration model."""

from pydantic import BaseModel


class TiledConfig(BaseModel):
    """Configuration for the Tiled connection."""
    url: str = ""
    token: str = ""
