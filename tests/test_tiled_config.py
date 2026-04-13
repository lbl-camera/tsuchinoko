"""Tests for Tiled configuration."""

from tsuchinoko.tiled.config import TiledConfig
from tsuchinoko.config import AppConfig


class TestTiledConfig:
    def test_defaults(self):
        cfg = TiledConfig()
        assert cfg.url == ""
        assert cfg.token == ""

    def test_custom(self):
        cfg = TiledConfig(url="https://tiled.example.com", token="jwt123")
        assert cfg.url == "https://tiled.example.com"
        assert cfg.token == "jwt123"

    def test_disabled_by_default(self):
        cfg = TiledConfig()
        assert not cfg.url

    def test_app_config_includes_tiled(self):
        cfg = AppConfig()
        assert hasattr(cfg, 'tiled')
        assert isinstance(cfg.tiled, TiledConfig)

    def test_app_config_from_dict(self):
        cfg = AppConfig(**{"tiled": {"url": "https://t.example.com", "token": "tok"}})
        assert cfg.tiled.url == "https://t.example.com"
