import sys
from functools import partial

from loguru import logger
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QKeySequence, QAction, QActionGroup, QShortcut
from PySide6.QtWidgets import QMenuBar, QMenu, QWidget, QApplication

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager


class DebuggableMenuBar(QMenuBar):
    def __init__(self, *args, **kwargs):
        super(DebuggableMenuBar, self).__init__(*args, **kwargs)

        self.debugshortcut = QShortcut(QKeySequence("Ctrl+Return"),
                                       self,
                                       self.showDebugMenu,
                                       context=Qt.ApplicationShortcut)

        self._debugmenu = QMenu("Debugging")
        self._debugmenu.addAction("Debug widget", self.startDebugging)
        self._loggingmenu = QMenu("Logging level")
        self._debugmenu.addMenu(self._loggingmenu)
        self._levels_group = QActionGroup(self)
        self._levels_group.setExclusive(True)
        for level in ['Critical', 'Error', 'Warning', 'Info', 'Debug']:
            action = QAction(level)
            action.triggered.connect(partial(self.set_level, level))
            action.setCheckable(True)
            self._levels_group.addAction(action)
            self._loggingmenu.addAction(action)
        self._loggingmenu.actions()[2].setChecked(True)

        self.mousedebugger = MouseDebugger()

    def set_level(self, level: str):
        from . import displays
        logger.remove(displays.log_handler_id)
        log_handler = displays.LogHandler.get_current()
        if log_handler:
            displays.log_handler_id = logger.add(log_handler, level=level.upper())
        logger.critical(f'Log level set to {level.upper()}')

    def showDebugMenu(self):
        self.addMenu(self._debugmenu)

    def startDebugging(self):
        QApplication.instance().installEventFilter(self.mousedebugger)


class MouseDebugger(QObject):
    def eventFilter(self, obj, event):
        # print(event,obj)
        # print(self.sender())
        if event.type() == QEvent.MouseButtonPress:
            print(QApplication.instance().activeWindow().childAt(event.pos()))
            IPythonDebugger(QApplication.instance().activeWindow().childAt(event.pos())).show()
            QApplication.instance().removeEventFilter(self)
            return True
        return False


class IPythonDebugger(RichJupyterWidget):
    def __init__(self, widget: QWidget):
        super(IPythonDebugger, self).__init__()

        # Setup the kernel
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        kernel = self.kernel_manager.kernel
        kernel.gui = "qt"

        # Push QWidget to the console
        kernel.shell.push({"widget": widget})

        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

        # Setup console widget
        def stop():
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()

        self.exit_requested.connect(stop)


if __name__ == "__main__":
    from PySide6.QtWidgets import QMainWindow, QLabel

    app = QApplication([])
    window = QMainWindow()
    window.setCentralWidget(QLabel("test"))
    db = DebuggableMenuBar()
    window.setMenuBar(db)
    window.show()

    app.exec_()
