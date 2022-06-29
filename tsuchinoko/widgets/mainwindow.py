import time

from loguru import logger
import numpy as np
from PySide2.QtCore import QTimer
from PySide2.QtGui import QIcon
from pyqtgraph import PlotWidget, ScatterPlotItem, CurveArrow, TextItem, mkBrush, mkColor, mkPen
from pyqtgraph.dockarea import DockArea
from qtmodern.styles import dark
from qtpy.QtWidgets import QMainWindow, QApplication

from tsuchinoko.adaptive import Data
from tsuchinoko.graphics_items.clouditem import CloudItem
from tsuchinoko.graphics_items.indicatoritem import BetterCurveArrow
from tsuchinoko.utils.threads import method, invoke_in_main_thread, iterator, QThreadFutureIterator
from tsuchinoko.widgets.displays import Log, Configuration, RunEngineControls, GraphManager


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle('Tsuchinoko')
        self.setWindowIcon(QIcon('assets/tsuchinoko.png'))
        self.resize(1700, 1000)

        self.log_widget = Log()
        self.configuration_widget = Configuration()
        # self.run_engine_widget = RunEngineControls()
        self.graph_manager_widget = GraphManager()

        self.dock_area = DockArea()
        self.setCentralWidget(self.dock_area)

        for position, w, *relaltive_to in [('bottom', self.graph_manager_widget),
                                           ('bottom', self.log_widget, self.graph_manager_widget),
                                           ('right', self.configuration_widget, self.graph_manager_widget),
                                           # ('bottom', self.run_engine_widget, self.configuration_widget),
                                           ]:
            self.dock_area.addDock(w, position, *relaltive_to)

        dark(QApplication.instance())

        self.init_socket()

        self.update_thread = QThreadFutureIterator(self.update, yield_slot=self.update_graphs)
        self.update_thread.start()


    def init_socket(self):
        import zmq
        context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to core server…")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        self.socket.RCVTIMEO = 5000

    def update(self):
        data = None
        import json, zmq

        while True:
            if data:
                message = f'partial_data {len(data)}'.encode()
            else:
                message = b"full_data"

            try:
                logger.info(f'request: {message}')
                self.socket.send(message)
                #  Get the reply.
                message = self.socket.recv()
            except zmq.ZMQError as ex:
                logger.exception(ex)
                self.init_socket()
                data = None  # wipeout data and get a full update next time
                time.sleep(1)
            else:

                # print("Received reply [ %s ]" % (message))

                if data:
                    data.extend(Data(**json.loads(message)))
                else:
                    data = Data(**json.loads(message))
                yield data

    def init_graph(self, name, indicator='maxvalue'):
        if name not in self.graph_manager_widget.graphs:
            graph = PlotWidget()
            # scatter = ScatterPlotItem(name='scatter', x=[0], y=[0], size=10, pen=mkPen(None), brush=mkBrush(255, 255, 255, 120))
            cloud = CloudItem(name='scatter', size=10)

            graph.addItem(cloud)
            if indicator:
                max_arrow = BetterCurveArrow(cloud.scatter, brush=mkBrush('r'))
                last_arrow = BetterCurveArrow(cloud.scatter, brush=mkBrush('w'))
                text = TextItem()
                graph.addItem(max_arrow)
                # graph.addItem(text)

            def _update_graph(data, indicator='maxvalue'):
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

                cloud.setData(x=x,
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
                last_arrow.setIndex(len(x)-1)
                # text.setText(f'Max: {v[max_index]:.2f} ({x[max_index]:.2f}, {y[max_index]:.2f})')
                # text.setPos(x[max_index], y[max_index])

            self.graph_manager_widget.register_graph(name, graph, _update_graph)

    def update_graphs(self, data):
        for metric_name in ['variance', 'score', *data.metrics]:
            self.init_graph(metric_name)

        self.graph_manager_widget.update(data)
        # x, y = zip(*data.positions)
        #
        # for metric_name in data.metrics:
        #     self._update_graph(metric_name, x, y, data.metrics[metric_name])
        #
        # self._update_graph('score', x, y, data.scores)
        # self._update_graph('variance', x, y, data.variances)
