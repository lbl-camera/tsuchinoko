from tsuchinoko.nats.nats_capability import VENDORED_FROM, derive_prefix


def test_vendored_marker():
    assert VENDORED_FROM == "lightfall_utils.ipc.capability v2"


def test_derive_prefix_vectors():
    assert derive_prefix("test-secret-0001") == "r5itqnjn2uhkfjylxqpf7fwjv7"
    assert derive_prefix("A" * 43) == "3dcidrblqoesmah3mp7u5jx4cx"
