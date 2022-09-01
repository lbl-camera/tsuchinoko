import sys
import os
from threading import Thread
import inspect

from pytestqt import qtbot
from qtpy import QtCore

from tsuchinoko.core import ConnectResponse
from tsuchinoko.widgets.mainwindow import MainWindow

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def test_simple(qtbot):
    from examples import server_demo

    main_window = MainWindow()
    main_window.show()

    server_thread = Thread(target=server_demo.core.main)
    server_thread.start()

    qtbot.addWidget(main_window)
    with qtbot.waitCallback() as cb:
        main_window.subscribe(cb, ConnectResponse)
    qtbot.mouseClick(main_window.state_manager_widget.start_pause_button, QtCore.Qt.LeftButton)
    qtbot.wait(10000)  # give it time to run a few iterations
    qtbot.waitForWindowShown(main_window)

    assert len(main_window.data) > 0
