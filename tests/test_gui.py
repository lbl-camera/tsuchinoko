import sys
import os
from threading import Thread
import inspect

from pytest import fixture
from qtpy.QtWidgets import QMessageBox
from pytestqt import qtbot
from qtpy import QtCore
from loguru import logger

from tsuchinoko.core import ConnectResponse, ZMQCore, CoreState
from tsuchinoko.widgets.mainwindow import MainWindow
from .test_core import random_engine, simple_execution_engine, image_data, image_func

# Disable logging to console when running tests
# NOTE: it seems there is a bug between loguru and pytest; pytest tries to log to a tempfile, but closes it when finished
# NOTE: if loguru has a backlog of messages
# logger.remove()


@fixture
def client_window(qtbot):
    logger.info('starting client window setup')
    main_window = MainWindow()
    main_window.show()
    logger.info('client window setup complete')
    # qtbot.addWidget(main_window)
    with qtbot.wait_exposed(main_window):
        yield main_window

    logger.info('teardown client window')
    with qtbot.waitCallback() as cb:
        main_window.update_thread.sigFinished.connect(cb)
        main_window.close()
    logger.info('client window teardown finished')


@fixture
def dialog_response_no(monkeypatch):
    # Suppress save dialog
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)


def test_simple(qtbot, dialog_response_no, random_engine, simple_execution_engine, client_window):

    core = ZMQCore()
    core.set_execution_engine(simple_execution_engine)
    core.set_adaptive_engine(random_engine)
    server_thread = Thread(target=core.main)
    server_thread.start()

    with qtbot.waitCallback() as cb:
        client_window.subscribe(cb, ConnectResponse)

    def button_enabled():
        assert client_window.state_manager_widget.start_pause_button.isEnabled()

    qtbot.waitUntil(button_enabled)
    qtbot.mouseClick(client_window.state_manager_widget.start_pause_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: len(client_window.data) > 0)

    qtbot.mouseClick(client_window.state_manager_widget.stop_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: client_window.state_manager_widget.state == CoreState.Inactive)

    core.exit()
    server_thread.join()

