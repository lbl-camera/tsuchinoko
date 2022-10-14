import sys
import os
from threading import Thread
import inspect

from pytest import fixture
from qtpy.QtWidgets import QMessageBox, QFileDialog
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
    with monkeypatch.context() as m:
        m.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.No)
        yield


@fixture
def client_and_server(qtbot, dialog_response_no, random_engine, simple_execution_engine, client_window):

    core = ZMQCore()
    core.set_execution_engine(simple_execution_engine)
    core.set_adaptive_engine(random_engine)

    with qtbot.waitCallback() as cb:
        client_window.subscribe(cb, ConnectResponse)
        server_thread = Thread(target=core.main)
        server_thread.start()
        if client_window.state_manager_widget.state != CoreState.Connecting:
            cb()

    def button_enabled():
        assert client_window.state_manager_widget.start_pause_button.isEnabled()

    qtbot.waitUntil(button_enabled)
    qtbot.mouseClick(client_window.state_manager_widget.start_pause_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: len(client_window.data) > 0)

    yield client_window, core

    qtbot.mouseClick(client_window.state_manager_widget.stop_button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: client_window.state_manager_widget.state == CoreState.Inactive)

    core.exit()
    server_thread.join()


def test_simple(client_and_server):
    client_window, core = client_and_server
    assert len(client_window.data) > 0


def test_save_restore(client_and_server, monkeypatch, tmp_path):
    client_window, core = client_and_server

    with monkeypatch.context() as m:
        m.setattr(QFileDialog, 'getSaveFileName', lambda filter: (tmp_path / 'test.yml', "YAML (*.yml)"))
        client_window.save_data()

    with monkeypatch.context() as m:
        m.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
        client_window.new_data()

    with monkeypatch.context() as m:
        m.setattr(QFileDialog, 'getOpenFileName', lambda filter: (tmp_path / 'test.yml', "YAML (*.yml)"))
        m.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
        client_window.open_data()

    with monkeypatch.context() as m:
        m.setattr(QFileDialog, 'getSaveFileName', lambda filter: (tmp_path / 'test_params.yml', "YAML (*.yml)"))
        client_window.save_parameters()

    with monkeypatch.context() as m:
        m.setattr(QFileDialog, 'getOpenFileName', lambda filter: (tmp_path / 'test_params.yml', "YAML (*.yml)"))
        client_window.open_parameters()
