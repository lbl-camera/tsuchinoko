"""Tests for NATS configuration."""

import pytest
from tsuchinoko.nats.config import NATSConfig
from tsuchinoko.config import AppConfig


class TestNATSConfig:
    def test_defaults(self):
        cfg = NATSConfig()
        assert cfg.url == ""
        assert cfg.lucid_prefix == "als.7011"
        assert cfg.app_name == "tsuchinoko"
        assert cfg.app_version == ""
        assert cfg.auth_timeout == 70.0
        assert cfg.connect_timeout == 5.0
        assert cfg.reconnect is True

    def test_custom_values(self):
        cfg = NATSConfig(url="nats://broker:4222", lucid_prefix="als.1234")
        assert cfg.url == "nats://broker:4222"
        assert cfg.lucid_prefix == "als.1234"

    def test_nats_disabled_by_default(self):
        cfg = NATSConfig()
        assert not cfg.url

    def test_app_config_includes_nats(self):
        cfg = AppConfig()
        assert hasattr(cfg, 'nats')
        assert isinstance(cfg.nats, NATSConfig)
        assert cfg.nats.url == ""

    def test_app_config_from_dict(self):
        cfg = AppConfig(**{"nats": {"url": "nats://localhost:4222", "lucid_prefix": "test.prefix"}})
        assert cfg.nats.url == "nats://localhost:4222"
        assert cfg.nats.lucid_prefix == "test.prefix"
