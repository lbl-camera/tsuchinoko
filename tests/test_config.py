"""Tests for configuration management."""
import pytest
from pydantic import ValidationError

from tsuchinoko.config import (
    AppConfig,
    NetworkConfig,
    UIConfig,
    CoreConfig,
    get_config,
    set_config,
    reset_config,
)


class TestNetworkConfig:
    def test_default_values(self):
        config = NetworkConfig()
        assert config.address == 'localhost'
        assert config.port == 5555
        assert config.socket_linger == 5
        assert config.recv_timeout_ms == 5000
        assert config.queue_timeout_s == 0.2

    def test_custom_values(self):
        config = NetworkConfig(address='192.168.1.1', port=6666)
        assert config.address == '192.168.1.1'
        assert config.port == 6666

    def test_port_validation_below_minimum(self):
        with pytest.raises(ValidationError):
            NetworkConfig(port=100)  # Below 1024

    def test_port_validation_above_maximum(self):
        with pytest.raises(ValidationError):
            NetworkConfig(port=70000)  # Above 65535

    def test_port_at_boundaries(self):
        # Test minimum valid port
        config_min = NetworkConfig(port=1024)
        assert config_min.port == 1024
        # Test maximum valid port
        config_max = NetworkConfig(port=65535)
        assert config_max.port == 65535

    def test_socket_linger_validation(self):
        # Valid value
        config = NetworkConfig(socket_linger=10)
        assert config.socket_linger == 10
        # Invalid negative value
        with pytest.raises(ValidationError):
            NetworkConfig(socket_linger=-1)

    def test_recv_timeout_validation(self):
        # Too low
        with pytest.raises(ValidationError):
            NetworkConfig(recv_timeout_ms=50)
        # Too high
        with pytest.raises(ValidationError):
            NetworkConfig(recv_timeout_ms=70000)
        # Valid boundary values
        config_min = NetworkConfig(recv_timeout_ms=100)
        assert config_min.recv_timeout_ms == 100
        config_max = NetworkConfig(recv_timeout_ms=60000)
        assert config_max.recv_timeout_ms == 60000

    def test_queue_timeout_validation(self):
        # Too low
        with pytest.raises(ValidationError):
            NetworkConfig(queue_timeout_s=0.001)
        # Too high
        with pytest.raises(ValidationError):
            NetworkConfig(queue_timeout_s=15.0)
        # Valid values
        config = NetworkConfig(queue_timeout_s=1.0)
        assert config.queue_timeout_s == 1.0


class TestUIConfig:
    def test_default_values(self):
        config = UIConfig()
        assert config.window_width == 1700
        assert config.window_height == 1000
        assert config.max_log_entries == 100

    def test_custom_values(self):
        config = UIConfig(window_width=800, window_height=600, max_log_entries=500)
        assert config.window_width == 800
        assert config.window_height == 600
        assert config.max_log_entries == 500

    def test_window_width_validation(self):
        with pytest.raises(ValidationError):
            UIConfig(window_width=200)  # Below 400

    def test_window_height_validation(self):
        with pytest.raises(ValidationError):
            UIConfig(window_height=100)  # Below 300

    def test_max_log_entries_validation(self):
        with pytest.raises(ValidationError):
            UIConfig(max_log_entries=5)  # Below 10
        with pytest.raises(ValidationError):
            UIConfig(max_log_entries=20000)  # Above 10000


class TestCoreConfig:
    def test_default_values(self):
        config = CoreConfig()
        assert config.sleep_for_fresh_data == 0.1
        assert config.checkpoint_interval == 0

    def test_custom_values(self):
        config = CoreConfig(sleep_for_fresh_data=0.5, checkpoint_interval=10)
        assert config.sleep_for_fresh_data == 0.5
        assert config.checkpoint_interval == 10

    def test_sleep_for_fresh_data_validation(self):
        with pytest.raises(ValidationError):
            CoreConfig(sleep_for_fresh_data=0.001)  # Too low
        with pytest.raises(ValidationError):
            CoreConfig(sleep_for_fresh_data=15.0)  # Too high

    def test_checkpoint_interval_validation(self):
        with pytest.raises(ValidationError):
            CoreConfig(checkpoint_interval=-1)  # Negative not allowed


class TestAppConfig:
    def test_default_creates_nested_configs(self):
        config = AppConfig()
        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.ui, UIConfig)
        assert isinstance(config.core, CoreConfig)

    def test_from_dict(self):
        data = {'network': {'address': 'remote.host', 'port': 7777}}
        config = AppConfig(**data)
        assert config.network.address == 'remote.host'
        assert config.network.port == 7777
        # Other values should still be defaults
        assert config.network.socket_linger == 5
        assert config.ui.window_width == 1700

    def test_full_dict_configuration(self):
        data = {
            'network': {
                'address': '10.0.0.1',
                'port': 8888,
                'socket_linger': 10,
                'recv_timeout_ms': 10000,
                'queue_timeout_s': 0.5,
            },
            'ui': {
                'window_width': 1200,
                'window_height': 800,
                'max_log_entries': 200,
            },
            'core': {
                'sleep_for_fresh_data': 0.2,
                'checkpoint_interval': 5,
            },
        }
        config = AppConfig(**data)
        assert config.network.address == '10.0.0.1'
        assert config.network.port == 8888
        assert config.ui.window_width == 1200
        assert config.core.checkpoint_interval == 5

    def test_nested_validation_propagates(self):
        # Invalid nested config should raise validation error
        with pytest.raises(ValidationError):
            AppConfig(network={'port': 100})  # Invalid port


class TestGlobalConfig:
    def setup_method(self):
        """Reset global config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset global config after each test."""
        reset_config()

    def test_get_config_creates_default(self):
        config = get_config()
        assert isinstance(config, AppConfig)
        assert config.network.port == 5555

    def test_get_config_returns_same_instance(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config_replaces_global(self):
        custom_config = AppConfig(network=NetworkConfig(port=9999))
        set_config(custom_config)
        config = get_config()
        assert config.network.port == 9999

    def test_reset_config(self):
        # Get a config
        config1 = get_config()
        # Reset and get a new one
        reset_config()
        config2 = get_config()
        # They should be different instances
        assert config1 is not config2
