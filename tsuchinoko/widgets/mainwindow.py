from PySide2.QtGui import QIcon
from pyqtgraph.dockarea import DockArea
from qtmodern.styles import dark
from qtpy.QtWidgets import QMainWindow, QApplication

from tsuchinoko.widgets.displays import Log, Configuration, RunEngineControls, GraphManager


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Tsuchinoko')
        self.setWindowIcon(QIcon('assets/tsuchinoko.png'))

        self.log_widget = Log()
        self.configuration_widget = Configuration()
        self.run_engine_widget = RunEngineControls()
        self.graph_manager_widget = GraphManager()

        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

        for position, w, *relaltive_to in [('bottom', self.graph_manager_widget),
                                           ('bottom', self.log_widget, self.graph_manager_widget),
                                           ('right', self.configuration_widget, self.graph_manager_widget),
                                           ('bottom', self.run_engine_widget, self.configuration_widget),
                                           ]:
            self.dock_area.addDock(w, position, *relaltive_to)

        dark(QApplication.instance())

    @property
    def experiment(self):
        return self.run_engine_widget.experiment

    @experiment.setter
    def experiment(self, value):
        self.run_engine_widget.experiment = value
