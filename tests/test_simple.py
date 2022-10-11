import sys
import os
from threading import Thread
import inspect

from qtpy.QtWidgets import QMessageBox
from pytestqt import qtbot
from qtpy import QtCore
from loguru import logger
from pytest import fixture

from tsuchinoko.core import ConnectResponse
from tsuchinoko.utils.runengine import get_run_engine
from tsuchinoko.widgets.mainwindow import MainWindow

# Disable logging to console when running tests
# NOTE: it seems there is a bug between loguru and pytest; pytest tries to log to a tempfile, but closes it when finished
# NOTE: if loguru has a backlog of messages
# logger.remove()

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


@fixture
def bluesky_core():
    logger.info('starting setup')
    from examples import server_demo
    server_thread = Thread(target=server_demo.core.main)
    server_thread.start()
    logger.info('setup complete')

    yield

    # ensure that runengine thread terminates before ending test
    logger.info('starting teardown')
    RE = get_run_engine()
    RE.RE.halt()
    RE.process_queue_thread.requestInterruption()
    RE.process_queue_thread.wait()

    server_demo.core.exit()
    server_thread.join()
    logger.info('teardown complete')


def test_simple(qtbot, monkeypatch, bluesky_core):

    main_window = MainWindow()
    main_window.show()

    qtbot.addWidget(main_window)

    # Suppress save dialog
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    with qtbot.wait_exposed(main_window):
        with qtbot.waitCallback() as cb:
            main_window.subscribe(cb, ConnectResponse)

        def button_enabled():
            assert main_window.state_manager_widget.start_pause_button.isEnabled()

        qtbot.waitUntil(button_enabled)
        qtbot.mouseClick(main_window.state_manager_widget.start_pause_button, QtCore.Qt.LeftButton)
        qtbot.wait(3000)  # give it time to run a few iterations

    assert len(main_window.data) > 0


    # main_window.close()
    qtbot.wait_signal(main_window.update_thread.sigFinished)

