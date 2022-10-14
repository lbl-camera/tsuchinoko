import sys
import os
from threading import Thread
import inspect

from qtpy.QtWidgets import QMessageBox
from pytestqt import qtbot
from qtpy import QtCore
from loguru import logger

from tsuchinoko.core import ConnectResponse, ZMQCore
from tsuchinoko.widgets.mainwindow import MainWindow
from .test_core import random_engine, simple_execution_engine, image_data, image_func

# Disable logging to console when running tests
# NOTE: it seems there is a bug between loguru and pytest; pytest tries to log to a tempfile, but closes it when finished
# NOTE: if loguru has a backlog of messages
# logger.remove()


def test_simple(qtbot, monkeypatch, random_engine, simple_execution_engine):

    core = ZMQCore()
    core.set_execution_engine(simple_execution_engine)
    core.set_adaptive_engine(random_engine)
    server_thread = Thread(target=core.main)
    server_thread.start()

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
        qtbot.wait(1000)  # give it time to run a few iterations

    assert len(main_window.data) > 0


    # main_window.close()
    qtbot.wait_signal(main_window.update_thread.sigFinished)

    core.exit()
    server_thread.join()

