import time
from collections import defaultdict
from queue import Queue, Empty
from typing import Any, Type, Union

try:
    from yaml import CLoader as Loader, CDumper as Dumper, dump, load
except ImportError:
    from yaml import Loader, Dumper

import zmq
from zmq.error import ZMQError
import numpy as np
from PySide2.QtGui import QIcon
from loguru import logger
from pyqtgraph import mkBrush, mkPen, HistogramLUTWidget, PlotItem
from pyqtgraph.dockarea import DockArea
from qtmodern.styles import dark
from qtpy.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QWidget, QMenuBar, QAction, QStyle, QFileDialog, QDialog, QMessageBox

from tsuchinoko.assets import path
from tsuchinoko.adaptive import Data
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import PauseRequest, StartRequest, GetParametersRequest, SetParameterRequest, PartialDataRequest, FullDataRequest, StopRequest, Message, StateRequest, StateResponse, GetParametersResponse, FullDataResponse, PartialDataResponse, MeasureRequest, \
    ConnectRequest, ConnectResponse, PushDataRequest, ExceptionResponse
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

        menubar = QMenuBar()
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

        self.context = zmq.Context()
        self.socket = None
        self.core_address = core_address
        self.init_socket()

        self.state_manager_widget.sigPause.connect(self.pause)
        self.state_manager_widget.sigStart.connect(self.start)
        self.state_manager_widget.sigStop.connect(self.stop)
        self.configuration_widget.sigPushParameter.connect(self.set_parameter)
        self.configuration_widget.sigRequestParameters.connect(self.request_parameters)
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

    def request_measure(self, pos):
        self.message_queue.put(MeasureRequest(pos))

    def request_parameters(self):
        self.message_queue.put(GetParametersRequest())

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

    def init_graph(self, name, item_key):
        if name not in self.graph_manager_widget.graphs:
            if item_key == 'clouditem':
                name, widget, update_callback = self.init_cloud(name)
            elif item_key == 'imageitem':
                name, widget, update_callback = self.init_image(name)
            else:
                logger.exception(ValueError(f'Invalid item key: {item_key}'))

        self.graph_manager_widget.register_graph(name, widget, update_callback)

    @staticmethod
    def init_image(name):
        graph = PlotItem()
        widget = ImageViewBlend(view=graph)
        graph.vb.invertY(False)  # imageview forces invertY; this resets it

        def _update_graph(data, last_data_size):
            if name in data.states:
                v = data.states[name]
                widget.imageItem.setImage(v, autoLevels=widget.imageItem.image is None)

        return name, widget, _update_graph

    @staticmethod
    def init_cloud(name):
        graph = ClickRequesterPlot()
        # scatter = ScatterPlotItem(name='scatter', x=[0], y=[0], size=10, pen=mkPen(None), brush=mkBrush(255, 255, 255, 120))
        cloud = CloudItem(name='scatter', size=10)
        histlut = HistogramLUTWidget()
        histlut.setImageItem(cloud)

        widget = QWidget()
        widget.setLayout(QHBoxLayout())

        widget.layout().addWidget(graph)
        widget.layout().addWidget(histlut)

        graph.addItem(cloud)

        # Hard-coded to show max
        max_arrow = BetterCurveArrow(cloud.scatter, brush=mkBrush('r'))
        last_arrow = BetterCurveArrow(cloud.scatter, brush=mkBrush('w'))
        # text = TextItem()
        graph.addItem(max_arrow)

        # graph.addItem(text)

        def _update_graph(data, last_data_size, indicator='maxvalue'):
            require_clear = False

            with data.r_lock():
                if name == 'Score':
                    v = data.scores.copy()
                elif name == 'Variance':
                    v = data.variances.copy()
                elif name in data.metrics:
                    v = data.metrics[name].copy()
                elif name in data.states:
                    v = data.states[name].copy()
                    require_clear = True
                else:
                    raise ValueError(f'No value for key: {name}')

                x, y = zip(*data.positions)

            lengths = len(v), len(x), len(y)

            if not np.all(np.array(lengths) == min(lengths)):
                logger.warning(f'Ragged arrays passed to cloud item with lengths (v, x, y): {lengths}')
                x = x[:min(lengths)]
                y = y[:min(lengths)]
                v = v[:min(lengths)]

            if not len(x):
                return

            # c = [255 * i / len(x) for i in range(len(x))]
            max_index = np.argmax(v)

            last_data_size = min(last_data_size, len(cloud.cData))

            if last_data_size == 0:
                action = cloud.setData
            elif require_clear:
                action = cloud.updateData
            else:
                action = cloud.extendData
                x = x[last_data_size + 1:]
                y = y[last_data_size + 1:]
                v = v[last_data_size + 1:]

            action(x=x,
                   y=y,
                   c=v,
                   data=v,
                   # size=5,
                   hoverable=True,
                   # hoverSymbol='s',
                   # hoverSize=6,
                   hoverPen=mkPen('b', width=2),
                   # hoverBrush=mkBrush('g'),
                   )
            # scatter.setData(
            #     [{'pos': (xi, yi),
            #       'size': (vi - min(v)) / (max(v) - min(v)) * 20 + 2 if max(v) != min(v) else 20,
            #       'brush': mkBrush(color=mkColor(255, 255, 255)) if i == len(x) - 1 else mkBrush(
            #           color=mkColor(255 - c, c, 0)),
            #       'symbol': '+' if i == len(x) - 1 else 'o'}
            #      for i, (xi, yi, vi, c) in enumerate(zip(x, y, v, c))])

            max_arrow.setIndex(max_index)
            last_arrow.setIndex(len(cloud.cData) - 1)
            # text.setText(f'Max: {v[max_index]:.2f} ({x[max_index]:.2f}, {y[max_index]:.2f})')
            # text.setPos(x[max_index], y[max_index])

        return name, widget, _update_graph

    def update_graphs(self, data, last_data_size):
        for metric_name in ['Variance', 'Score', *data.metrics, *data.states]:
            if metric_name not in self.graph_manager_widget.graphs:
                self.init_graph(metric_name, data.graphics_items.get(metric_name, 'clouditem'))

        self.graph_manager_widget.update(data, last_data_size)
        # x, y = zip(*data.positions)
        #
        # for metric_name in data.metrics:
        #     self._update_graph(metric_name, x, y, data.metrics[metric_name])
        #
        # self._update_graph('score', x, y, data.scores)
        # self._update_graph('variance', x, y, data.variances)

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
        self.graph_manager_widget.clear()
        self.update_graphs(self.data, 0)
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
        dump(state, open(name, 'r'), Dumper=Dumper)

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
        self.graph_manager_widget.clear()

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
