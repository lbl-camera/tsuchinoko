import time
from collections import defaultdict
from functools import partial
from queue import Queue, Empty
import subprocess
from signal import SIGINT
from typing import Any, Type, Union
import sys
from pathlib import Path

from tsuchinoko.utils.dependencies import check_dependencies
from tsuchinoko.widgets.debugmenubar import DebuggableMenuBar
from tsuchinoko.widgets.server_editor import ServerEditor

try:
    from yaml import CLoader as Loader, CDumper as Dumper, dump, load
except ImportError:
    from yaml import Loader, Dumper

import zmq
from zmq.error import ZMQError
import numpy as np
from qtpy.QtGui import QIcon
from loguru import logger
from pyqtgraph import mkBrush, mkPen, HistogramLUTWidget, PlotItem
from pyqtgraph.dockarea import DockArea
from qtmodern.styles import dark
from qtpy.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QWidget, QMenuBar, QAction, QStyle, QFileDialog, QDialog, QMessageBox

from tsuchinoko.assets import path
from tsuchinoko.adaptive import Data
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import PauseRequest, StartRequest, GetParametersRequest, SetParameterRequest, \
    PartialDataRequest, FullDataRequest, StopRequest, Message, StateRequest, StateResponse, GetParametersResponse, \
    FullDataResponse, PartialDataResponse, MeasureRequest, \
    ConnectRequest, ConnectResponse, PushDataRequest, ExceptionResponse, PullGraphsRequest, GraphsResponse, \
    ReplayRequest, ExitRequest, PushGraphsRequest
from tsuchinoko.graphics_items.clouditem import CloudItem
from tsuchinoko.graphics_items.indicatoritem import BetterCurveArrow
from tsuchinoko.graphics_items.mixins import ClickRequester, request_relay, ClickRequesterPlot
from tsuchinoko.utils.threads import QThreadFutureIterator, invoke_as_event
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
        open_server_editor = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), 'Open Server Editor', parent=file_menu)
        file_menu.addAction(open_data_action)
        file_menu.addAction(open_parameters_action)
        file_menu.addAction(save_data_action)
        file_menu.addAction(save_parameters_action)
        file_menu.addAction(open_server_editor)
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
        open_server_editor.triggered.connect(self.open_server_editor)

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

        self.context = zmq.Context()
        self.socket = None
        self.core_address = core_address
        self.init_socket()

        self.state_manager_widget.sigPause.connect(self.pause)
        self.state_manager_widget.sigStart.connect(self.start)
        self.state_manager_widget.sigStop.connect(self.stop)
        self.state_manager_widget.sigReplay.connect(self.replay)
        self.configuration_widget.sigPushParameter.connect(self.set_parameter)
        self.configuration_widget.sigRequestParameters.connect(self.request_parameters)
        self.graph_manager_widget.sigPush.connect(self.push_graph)
        request_relay.sigRequestMeasure.connect(self.request_measure)

        self.update_thread = QThreadFutureIterator(self.update, finished_slot=self.close_zmq, name='tsuchinoko-update')
        self.update_thread.start()

        self.data: Data = Data()
        self.last_data_size = None
        self.callbacks = defaultdict(list)

        self.subscribe(self.state_manager_widget.update_state, StateResponse)
        self.subscribe(self.state_manager_widget.update_state, ConnectResponse)
        self.subscribe(self.configuration_widget.update_parameters, GetParametersResponse, invoke_as_event=True)
        self.subscribe(self._data_callback, FullDataResponse)
        self.subscribe(self._data_callback, PartialDataResponse)
        self.subscribe(self.refresh_state, ConnectResponse)
        self.subscribe(self.log_widget.log_exception, ExceptionResponse)
        self.subscribe(self.set_graphs, GraphsResponse, invoke_as_event=True)

        self._server = None

    def init_socket(self):
        if self.socket:
            logger.debug("Closing socket")
            self.socket.close()

        #  Socket to talk to server
        logger.info("Connecting to core server…")
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 5)
        self.socket.connect(f"tcp://{self.core_address}:5555")
        self.socket.RCVTIMEO = 5000
        self.message_queue = Queue()

    def try_connect(self):
        self.message_queue.put(ConnectRequest())

    def get_state(self):
        self.message_queue.put(StateRequest())

    def pause(self):
        self.message_queue.put(PauseRequest())

    def start(self):
        self.message_queue.put(StartRequest())

    def stop(self):
        self.message_queue.put(StopRequest())

    def replay(self):
        message = ReplayRequest(self.data.positions, self.data.measurements)
        self.message_queue.put(StopRequest())
        self.message_queue.put(message)
        self.message_queue.put(StartRequest())

    def request_measure(self, pos):
        self.message_queue.put(MeasureRequest(pos))

    def request_parameters(self):
        self.message_queue.put(GetParametersRequest())

    def push_graph(self, graph):
        self.message_queue.put(PushGraphsRequest([graph]))

    def set_parameter(self, child_path: str, value: Any):
        if child_path:
            self.message_queue.put(SetParameterRequest(child_path, value))

    def update(self):
        self.data = Data()
        self.last_data_size = 0

        while True:
            yield
            request = None

            if self.state_manager_widget.state == CoreState.Stopping:
                self.last_data_size = 0
                self.data = Data()

            if self.state_manager_widget.state == CoreState.Connecting:
                self.try_connect()

            if self.state_manager_widget.state in [CoreState.Pausing, CoreState.Starting, CoreState.Resuming, CoreState.Resuming, CoreState.Stopping]:
                self.get_state()

            try:
                request = self.message_queue.get(timeout=.2)
            except Empty:
                if self.state_manager_widget.state == CoreState.Running:
                    if self.data:
                        request = PartialDataRequest(len(self.data))
                    else:
                        request = FullDataRequest()

            if not request:
                self.get_state()
                continue
            else:
                logger.info(f'request: {request}')

            try:
                self.socket.send_pyobj(request)
                response = self.socket.recv_pyobj()
            except ZMQError as ex:
                logger.warning(f'Unable to connect to core server at {self.core_address}...')
                time.sleep(1)
                # logger.exception(ex)
                if self.context:
                    self.init_socket()
                self.data = Data()  # wipeout data and get a full update next time
                self.last_data_size = 0
                self.state_manager_widget.update_state(CoreState.Connecting)
            else:
                logger.info(f'response: {response}')
                if not response:
                    self.get_state()
                else:
                    for callback, as_event in self.callbacks[type(response)]:
                        if as_event:
                            invoke_as_event(callback, *response.payload)
                        else:
                            callback(*response.payload)

    def _data_callback(self, data_payload, last_data_size=None):
        if not isinstance(data_payload, dict):  # TODO: Remove when responses are mapped to callbacks
            return

        if last_data_size is not None and last_data_size < len(self.data):
            raise IndexError('Overwriting of previous data prevented.')

        self.data.extend(Data(**data_payload))
        if len(data_payload['positions']):
            invoke_as_event(self.update_graphs, self.data, self.last_data_size)
        self.last_data_size = len(self.data)

    def refresh_state(self, _):
        self.message_queue.queue.clear()
        self.message_queue.put(GetParametersRequest())
        self.message_queue.put(PullGraphsRequest())

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
        self.callbacks[response_type].append((callback, invoke_as_event))

    def unsubscribe(self, callback, response_type: Union[Type[Message], None] = None):
        self.callbacks[response_type] = list(filter(lambda match_callback, invoke_as_event: match_callback == callback, self.callbacks[response_type]))

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
        # self.graph_manager_widget.reset()
        self.message_queue.put(PullGraphsRequest())
        # self.update_graphs(self.data, 0)
        if self.state_manager_widget.state == CoreState.Connecting:
            logger.warning('Data has been loaded before connecting to an experiment server. Remember to reload data after a connection is established.')
        else:
            result = QMessageBox.question(self,
                                          'Send data to server?',
                                          "Would you like to send the opened data to the experiment server? This will overwrite the server's current data.",
                                          buttons=QMessageBox.StandardButtons(QMessageBox.Yes | QMessageBox.No),
                                          defaultButton=QMessageBox.Yes)
            if result == QMessageBox.Yes:
                self.message_queue.put(PushDataRequest(self.data.as_dict()))

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

    def close_zmq(self):
        if self.update_thread.running:
            logger.info('waiting for update thread to finish')
            self.update_thread.requestInterruption()
            self.update_thread.wait()
        if self.socket:
            logger.debug('Closing socket')
            self.socket.close()
            self.socket = None
        if self.context:
            logger.debug('Closing context')
            self.context.term()
            self.context = None

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
                self.close_zmq()
            else:
                event.ignore()
        else:
            event.accept()
            self.close_zmq()

    def start_demo(self, demo_key):
        # If not communicating with localhost, dump connection to server
        if self.core_address != 'localhost':
            self.core_address = 'localhost'
            self.init_socket()

        # if there's a child process server, exit it
        if not self.close_demo(confirm=True):
            return

        suffix = Path(sys.executable).suffix 
        demo_exe = (Path(sys.executable).parent/'tsuchinoko_demo').with_suffix(suffix if suffix=='.exe' else '')
        # print(demo_exe)
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
            self.message_queue.put(ExitRequest())
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

    def open_server_editor(self):
        self.editor_window = ServerEditor(main_window=self)
        self.editor_window.show()
