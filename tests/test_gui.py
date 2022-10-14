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
def client_window():
    main_window = MainWindow()
    main_window.show()
    # qtbot.addWidget(main_window)
    yield main_window

    client_window.close()
    qtbot.wait_signal(client_window.update_thread.sigFinished)


def test_simple(qtbot, monkeypatch, random_engine, simple_execution_engine, client_window):

    core = ZMQCore()
    core.set_execution_engine(simple_execution_engine)
    core.set_adaptive_engine(random_engine)
    server_thread = Thread(target=core.main)
    server_thread.start()

    # Suppress save dialog
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)

    with qtbot.wait_exposed(client_window):
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

