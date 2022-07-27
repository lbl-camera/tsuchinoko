import time
from collections import defaultdict
from queue import Queue, Empty
from typing import Any, Type, Union

import numpy as np
from PySide2.QtGui import QIcon
from loguru import logger
from pyqtgraph import PlotWidget, TextItem, mkBrush, mkPen, HistogramLUTWidget
from pyqtgraph.dockarea import DockArea
from qtmodern.styles import dark
from qtpy.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QWidget
from zmq import ZMQError

from tsuchinoko.adaptive import Data
from tsuchinoko.core import CoreState
from tsuchinoko.core.messages import StateRequest, PauseRequest, StartRequest, GetParametersRequest, SetParameterRequest, PartialDataRequest, FullDataRequest, StopRequest, Message, StateRequest, StateResponse, GetParametersResponse, FullDataResponse, PartialDataResponse
from tsuchinoko.graphics_items.clouditem import CloudItem
from tsuchinoko.graphics_items.indicatoritem import BetterCurveArrow
from tsuchinoko.utils.threads import QThreadFutureIterator, invoke_as_event, invoke_in_main_thread
from tsuchinoko.widgets.displays import Log, Configuration, GraphManager, StateManager


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Tsuchinoko')
        self.setWindowIcon(QIcon('assets/tsuchinoko.png'))
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

        self.init_socket()

        self.state_manager_widget.sigPause.connect(self.pause)
        self.state_manager_widget.sigStart.connect(self.start)
        self.state_manager_widget.sigStop.connect(self.stop)
        self.configuration_widget.sigPushParameter.connect(self.set_parameter)
        self.configuration_widget.sigRequestParameters.connect(self.request_parameters)

        self.update_thread = QThreadFutureIterator(self.update)
        self.update_thread.start()

        self.data: Data = Data()
        self.last_data_size = None
        self.callbacks = defaultdict(list)

        self.subscribe(self.state_manager_widget.update_state, StateResponse)
        self.subscribe(self.configuration_widget.update_parameters, GetParametersResponse)
        self.subscribe(self._data_callback, FullDataResponse)
        self.subscribe(self._data_callback, PartialDataResponse)

    def init_socket(self):
        import zmq
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to core server…")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        self.socket.RCVTIMEO = 5000
        self.message_queue = Queue()

    def get_state(self):
        self.message_queue.put(StateRequest())

    def pause(self):
        self.message_queue.put(PauseRequest())

    def start(self):
        self.message_queue.put(StartRequest())

    def stop(self):
        self.message_queue.put(StopRequest())

    def request_parameters(self):
        self.message_queue.put(GetParametersRequest())

    def set_parameter(self, child_path: str, value: Any):
        if child_path:
            self.message_queue.put(SetParameterRequest(child_path, value))

    def update(self):
        self.data = Data()
        self.last_data_size = 0
        import json, zmq

        while True:
            request = None

            if self.state_manager_widget.state == CoreState.Stopping:
                self.last_data_size = 0

            if self.state_manager_widget.state in [CoreState.Connecting, CoreState.Pausing, CoreState.Starting, CoreState.Resuming, CoreState.Resuming, CoreState.Stopping]:
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
                logger.info('Unable to connect to core server...')
                time.sleep(1)
                logger.exception(ex)
                self.init_socket()
                self.data = Data()  # wipeout data and get a full update next time
                self.last_data_size = 0
            else:
                logger.info(f'response: {response}')
                if not response:
                    self.get_state()
                else:
                    for callback in self.callbacks[type(response)]:
                        if callback == self._data_callback:
                            callback(*response.payload)
                        else:
                            invoke_in_main_thread(callback, *response.payload)

    def _data_callback(self, data_payload, last_data_size=None):
        if not isinstance(data_payload, dict):  # TODO: Remove when responses are mapped to callbacks
            return

        if last_data_size is not None and last_data_size < len(self.data):
            raise IndexError('Overwriting of previous data prevented.')

        self.data.extend(Data(**data_payload))
        if len(data_payload['positions']):
            invoke_as_event(self.update_graphs, self.data, self.last_data_size)
        self.last_data_size = len(self.data)

    def init_graph(self, name, indicator='maxvalue'):
        if name not in self.graph_manager_widget.graphs:
            graph = PlotWidget()
            # scatter = ScatterPlotItem(name='scatter', x=[0], y=[0], size=10, pen=mkPen(None), brush=mkBrush(255, 255, 255, 120))
            cloud = CloudItem(name='scatter', size=10)
            histlut = HistogramLUTWidget()
            histlut.setImageItem(cloud)

            widget = QWidget()
            widget.setLayout(QHBoxLayout())

            widget.layout().addWidget(graph)
            widget.layout().addWidget(histlut)


            graph.addItem(cloud)
            if indicator:
                max_arrow = BetterCurveArrow(cloud.scatter, brush=mkBrush('r'))
                last_arrow = BetterCurveArrow(cloud.scatter, brush=mkBrush('w'))
                text = TextItem()
                graph.addItem(max_arrow)
                # graph.addItem(text)

            def _update_graph(data, last_data_size, indicator='maxvalue'):
                with data:
                    if name == 'score':
                        v = data.scores
                    elif name == 'variance':
                        v = data.variances
                    else:
                        v = data.metrics[name]

                    x, y = zip(*data.positions)

                # c = [255 * i / len(x) for i in range(len(x))]
                max_index = np.argmax(v)

                if last_data_size == 0:
                    action = cloud.setData
                else:
                    action = cloud.extendData

                action(x=x[last_data_size:],
                       y=y[last_data_size:],
                       c=v[last_data_size:],
                       data=v[last_data_size:],
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
                last_arrow.setIndex(len(x)-1)
                # text.setText(f'Max: {v[max_index]:.2f} ({x[max_index]:.2f}, {y[max_index]:.2f})')
                # text.setPos(x[max_index], y[max_index])

            self.graph_manager_widget.register_graph(name, widget, _update_graph)

    def update_graphs(self, data, last_data_size):
        for metric_name in ['variance', 'score', *data.metrics]:
            self.init_graph(metric_name)

        self.graph_manager_widget.update(data, last_data_size)
        # x, y = zip(*data.positions)
        #
        # for metric_name in data.metrics:
        #     self._update_graph(metric_name, x, y, data.metrics[metric_name])
        #
        # self._update_graph('score', x, y, data.scores)
        # self._update_graph('variance', x, y, data.variances)

    def subscribe(self, callback, response_type:Union[Type[Message], None] = None):
        self.callbacks[response_type].append(callback)

    def unsubscribe(self, callback, response_type:Union[Type[Message], None] = None):
        self.callbacks[response_type].remove(callback)