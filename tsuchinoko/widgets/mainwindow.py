import pyqtgraph
from qtmodern.styles import dark
from pyqtgraph.dockarea import DockArea
from qtpy.QtWidgets import QMainWindow, QApplication

from tsuchinoko.widgets.displays import Log, Configuration, RunEngineControls, GraphManager


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

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
