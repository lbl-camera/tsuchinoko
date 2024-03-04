import time


def test_core(core):
    time.sleep(1)
    assert len(core.data)

