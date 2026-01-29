import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any, Type, Union

from tsuchinoko.utils.dependencies import check_dependencies
from tsuchinoko.widgets.debugmenubar import DebuggableMenuBar

try:
    from yaml import CLoader as Loader, CDumper as Dumper, dump, load
except ImportError:
    from yaml import Loader, Dumper

from PySide6.QtGui import QIcon, QAction
from loguru import logger
from pyqtgraph.dockarea import DockArea
from qtmodern.styles import dark
from PySide6.QtWidgets import QMainWindow, QApplication, QStyle, QFileDialog, QMessageBox

from tsuchinoko.assets import path
from tsuchinoko.adaptive import Data
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import (
    Message, StateResponse, GetParametersResponse, FullDataResponse,
    PartialDataResponse, ConnectResponse, ExceptionResponse, GraphsResponse
)
from tsuchinoko.graphics_items.mixins import ClickRequester, request_relay
from tsuchinoko.network import NetworkManager
from tsuchinoko.utils.threads import invoke_as_event
from tsuchinoko.widgets.displays import Log, Configuration, GraphManager, StateManager


class ImageViewBlend(ClickRequester):
    pass


class MainWindow(QMainWindow):
    def __init__(self, core_address='localhost'):
        super(MainWindow, self).__init__()

        menubar = DebuggableMenuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction('&New...', self.new_data)
        open_data_action = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), 'Open data...', parent=file_menu)
        open_parameters_action = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), 'Open parameters...', parent=file_menu)
        save_data_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save data as...', parent=file_menu)
        save_parameters_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), 'Save parameters as...', parent=file_menu)
        file_menu.addAction(open_data_action)
        file_menu.addAction(open_parameters_action)
        file_menu.addAction(save_data_action)
        file_menu.addAction(save_parameters_action)
        file_menu.addAction('E&xit', self.close)
        self.setMenuBar(menubar)

        demo_menu = menubar.addMenu("&Demo")
        demo_menu.addAction('Start &Simple Demo', partial(self.start_demo, 'server_demo'))
        demo_menu.addAction('Start &Adaptive Demo', partial(self.start_demo, 'adaptive_demo')).setEnabled(check_dependencies(['adaptive']))
        demo_menu.addAction('Start &Grid Scan Demo', partial(self.start_demo, 'grid_demo'))
        demo_menu.addAction('Start &High Dimensionality Demo',  partial(self.start_demo, 'high_dimensionality_server_demo')).setEnabled(check_dependencies(['perlin-noise']))
        demo_menu.addAction('Start &Multi Task Demo',  partial(self.start_demo, 'multi_task_server_demo'))
        demo_menu.addAction('Start &Quad Tree Demo',  partial(self.start_demo, 'quadtree_demo'))
        demo_menu.addAction('Start &Bluesky Demo',  partial(self.start_demo, 'server_demo_bluesky'))

        save_data_action.triggered.connect(self.save_data)
        open_data_action.triggered.connect(self.open_data)
        save_parameters_action.triggered.connect(self.save_parameters)
        open_parameters_action.triggered.connect(self.open_parameters)

        self.setWindowTitle('Tsuchinoko')
        self.setWindowIcon(QIcon(path('tsuchinoko.png')))
        self.resize(1700, 1000)

        self.log_widget = Log()
        self.configuration_widget = Configuration()
        self.state_manager_widget = StateManager()
        self.graph_manager_widget = GraphManager()

        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

        for position, w, *relaltive_to in [('bottom', self.graph_manager_widget),
                                           ('bottom', self.log_widget, self.graph_manager_widget),
                                           ('right', self.configuration_widget, self.graph_manager_widget),
                                           ('top', self.state_manager_widget, self.configuration_widget),
                                           ]:
            self.dock_area.addDock(w, position, *relaltive_to)

        dark(QApplication.instance())

        # Initialize data
        self.data: Data = Data()
        self.last_data_size = 0

        # Initialize network manager
        self.core_address = core_address
        self.network = NetworkManager(address=core_address)
        self.network.set_state_getter(lambda: self.state_manager_widget.state)
        self.network.set_on_connection_lost(self._on_connection_lost)
        self.network._create_data_request = self._create_data_request

        # Connect widget signals to network operations
        self.state_manager_widget.sigPause.connect(self.network.pause)
        self.state_manager_widget.sigStart.connect(self.network.start_experiment)
        self.state_manager_widget.sigStop.connect(self.network.stop)
        self.state_manager_widget.sigReplay.connect(self._replay)
        self.state_manager_widget.sigSetComputeMetrics.connect(self.network.set_compute_metrics)
        self.configuration_widget.sigPushParameter.connect(self.network.set_parameter)
        self.configuration_widget.sigRequestParameters.connect(self.network.request_parameters)
        self.graph_manager_widget.sigPush.connect(self.network.push_graph)
        request_relay.sigRequestMeasure.connect(self.network.request_measure)

        # Register response callbacks
        self.network.subscribe(self.state_manager_widget.update_state, StateResponse)
        self.network.subscribe(self.state_manager_widget.update_state, ConnectResponse)
        self.network.subscribe(self.configuration_widget.update_parameters, GetParametersResponse, invoke_as_event=True)
        self.network.subscribe(partial(self._data_callback, response_type='full'), FullDataResponse)
        self.network.subscribe(partial(self._data_callback, response_type='partial'), PartialDataResponse)
        self.network.subscribe(self._refresh_state, ConnectResponse)
        self.network.subscribe(self.log_widget.log_exception, ExceptionResponse)
        self.network.subscribe(self.set_graphs, GraphsResponse, invoke_as_event=True)

        # Start network communication
        self.network.start(finished_slot=self._close_network)

        self._server = None

    @property
    def update_thread(self):
        """Access the network update thread. For backward compatibility."""
        return self.network._update_thread

    def _replay(self):
        """Send replay request with current data."""
        self.network.replay(self.data.positions, self.data.measurements)

    def _create_data_request(self):
        """Create appropriate data request based on current data state."""
        from tsuchinoko.core.messages import PartialDataRequest, FullDataRequest
        if self.data:
            return PartialDataRequest(len(self.data))
        return FullDataRequest()

    def _on_connection_lost(self):
        """Handle connection loss - reset data and update UI."""
        self.data = Data()
        self.last_data_size = 0
        self.state_manager_widget.update_state(CoreState.Connecting, True)

    def _data_callback(self, data_payload, last_data_size=None, response_type='partial'):
        if not isinstance(data_payload, dict):  # TODO: Remove when responses are mapped to callbacks
            return

        if last_data_size is not None and last_data_size < len(self.data):
            raise IndexError('Overwriting of previous data prevented.')

        if response_type == 'full':
            self.data = Data(**data_payload)
            self.last_data_size = 0
        elif response_type == 'partial':
            self.data.extend(Data(**data_payload))
        else:
            raise ValueError()

        if len(data_payload['positions']):
            # stash the length of new data early to avoid events getting confused when pausing this thread
            old_last_data_size, self.last_data_size = self.last_data_size, len(self.data)
            invoke_as_event(self.update_graphs, self.data, old_last_data_size)

    def _refresh_state(self, _, __):
        """Refresh parameters and graphs after connection."""
        self.network.clear_queue()
        self.network.request_parameters()
        self.network.pull_graphs()

    def update_graphs(self, data, last_data_size):
        self.graph_manager_widget.update_graphs(data, last_data_size)
        # x, y = zip(*data.positions)
        #
        # for metric_name in data.metrics:
        #     self._update_graph(metric_name, x, y, data.metrics[metric_name])
        #
        # self._update_graph('score', x, y, data.scores)
        # self._update_graph('variance', x, y, data.variances)

    def set_graphs(self, graphs):
        self.graph_manager_widget.set_graphs(graphs, self.data)

    def subscribe(self, callback, response_type: Union[Type[Message], None] = None, invoke_as_event: bool = False):
        """Subscribe to network responses. Delegates to NetworkManager."""
        self.network.subscribe(callback, response_type, invoke_as_event)

    def unsubscribe(self, callback, response_type: Union[Type[Message], None] = None):
        """Unsubscribe from network responses. Delegates to NetworkManager."""
        self.network.unsubscribe(callback, response_type)

    def open_data(self):
        name, filter = QFileDialog.getOpenFileName(filter=("YAML (*.yml)"))
        if not name:
            return

        if len(self.data):
            result = QMessageBox.question(self,
                                          'Clear current data?',
                                          "Loading data will clear the current data set. Would you like to proceed?",
                                          buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.Cancel),
                                          defaultButton=QMessageBox.Yes)
            if result != QMessageBox.Yes:
                return

        self.data = Data(**load(open(name, 'r'), Loader=Loader))
        self.last_data_size = len(self.data)
        self.network.pull_graphs()
        if self.state_manager_widget.state == CoreState.Connecting:
            logger.warning('Data has been loaded before connecting to an experiment server. Remember to reload data after a connection is established.')
        else:
            result = QMessageBox.question(self,
                                          'Send data to server?',
                                          "Would you like to send the opened data to the experiment server? This will overwrite the server's current data.",
                                          buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.No),
                                          defaultButton=QMessageBox.Yes)
            if result == QMessageBox.Yes:
                self.network.push_data(self.data.as_dict())

    def save_data(self):
        name, filter = QFileDialog.getSaveFileName(filter=("YAML (*.yml)"))
        if not name:
            return
        with self.data.r_lock():
            dump(self.data.as_dict(), open(name, 'w'), Dumper=Dumper)

    def open_parameters(self):
        name, filter = QFileDialog.getOpenFileName(filter=("YAML (*.yml)"))
        if not name:
            return
        state = load(open(name, 'r'), Loader=Loader)
        self.configuration_widget.parameter.restoreState(state, addChildren=True, removeChildren=True)

    def save_parameters(self):
        name, filter = QFileDialog.getSaveFileName(filter=("YAML (*.yml)"))
        if not name:
            return
        state = self.configuration_widget.parameter.saveState(filter='user')
        dump(state, open(name, 'w'), Dumper=Dumper)

    def new_data(self):
        if len(self.data):
            result = QMessageBox.question(self,
                                          'Clear current data?',
                                          "There is an active data set in memory. Would you like to proceed with clearing the current data?",
                                          buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.Cancel),
                                          defaultButton=QMessageBox.Yes)
            if result != QMessageBox.Yes:
                return

        self.data = Data()
        self.graph_manager_widget.reset()

    def _close_network(self):
        """Clean up network resources."""
        self.network.close()

    def closeEvent(self, event):
        if not self.close_demo(confirm=True):
            event.ignore()
            return
        if self.data and len(self.data):
            result = QMessageBox.question(self,
                                      'Save data?',
                                      "You have unsaved data in the active experiment. Do you want to save the data?",
                                      buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel),
                                      defaultButton=QMessageBox.Yes)
            if result == QMessageBox.Yes:
                self.save_data()
            if result in [QMessageBox.Yes, QMessageBox.No]:
                event.accept()
                self._close_network()
            else:
                event.ignore()
        else:
            event.accept()
            self._close_network()

    def start_demo(self, demo_key):
        # If not communicating with localhost, reconnect to localhost
        if self.core_address != 'localhost':
            self.core_address = 'localhost'
            self.network.address = 'localhost'
            self.network.init_socket()

        # if there's a child process server, exit it
        if not self.close_demo(confirm=True):
            return

        suffix = Path(sys.executable).suffix
        demo_exe = (Path(sys.executable).parent/'tsuchinoko_demo').with_suffix(suffix if suffix=='.exe' else '')
        self._server = subprocess.Popen([demo_exe, demo_key])

    def start_server(self, path):
        suffix = Path(sys.executable).suffix
        bootstrap_exe = (Path(sys.executable).parent / 'tsuchinoko_bootstrap').with_suffix(suffix if suffix == '.exe' else '')
        # print(bootstrap_exe)
        self._server = subprocess.Popen([bootstrap_exe, path])

    def close_demo(self, confirm=False):
        if self._server:
            if confirm:
                result = QMessageBox.question(self,
                                              'Shutdown server?',
                                              "A Tsuchinoko experiment server is currently running. Would you like to stop the server?",
                                              buttons=QMessageBox.StandardButtons(
                                                  QMessageBox.Yes | QMessageBox.Cancel),
                                              defaultButton=QMessageBox.Yes)
                if result != QMessageBox.Yes:
                    return False
            self.network.request_exit()
            try:
                self._server.wait(3)

            except subprocess.TimeoutExpired:
                result = QMessageBox.question(self,
                                              'Server not responding.',
                                              "The currently running demo server is not responding. Would you like to terminate it?",
                                              buttons=QMessageBox.StandardButtons(
                                                  QMessageBox.Yes | QMessageBox.Cancel),
                                              defaultButton=QMessageBox.Yes)
                if result == QMessageBox.Yes:
                    self._server.kill()
                    return True
                elif result == QMessageBox.Cancel:
                    return False
        return True

    stop_server = close_demo
