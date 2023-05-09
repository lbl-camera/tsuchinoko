from typing import List, Any, Tuple
import logging

from qtpy.QtCore import QObject, Signal, Qt
from qtpy.QtGui import QBrush
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter
from qtpy.QtWidgets import QFormLayout, QWidget, QListWidget, QListWidgetItem, QPushButton, QLabel, QSpacerItem, QSizePolicy, QStyle, QToolButton, QHBoxLayout, QVBoxLayout
from loguru import logger

from tsuchinoko.core import CoreState, ExceptionResponse
from tsuchinoko.graphs import graph_signal_relay
from tsuchinoko.utils import runengine
from tsuchinoko.utils.threads import invoke_as_event, invoke_in_main_thread


class Singleton(type(QObject)):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Display(Dock):
    ...


class LogHandler(logging.Handler):
    colors = {logging.DEBUG: Qt.gray, logging.ERROR: Qt.darkRed, logging.CRITICAL: Qt.red,
              logging.INFO: Qt.white, logging.WARNING: Qt.yellow}

    def __init__(self, log_widget, level=logging.WARNING):
        super(LogHandler, self).__init__(level=level)
        logging.getLogger().addHandler(self)
        self.log_widget = log_widget

        logger.add(logging.getLogger().handlers[-1], level=level)

    # follows same design as vanilla logger emissions
    def emit(self, record, level=logging.INFO, timestamp=None, icon=None, *args):  # We can have icons!
        item = QListWidgetItem(record.getMessage())
        item.setForeground(QBrush(self.colors[record.levelno]))
        item.setToolTip(timestamp)
        self.log_widget.insertItem(0, item)

        while self.log_widget.count() > 100:
            self.log_widget.takeItem(self.log_widget.count() - 1)

    def sink(self, message:str):
        item = QListWidgetItem(message.strip())
        self.log_widget.insertItem(0, item)

        while self.log_widget.count() > 100:
            self.log_widget.takeItem(self.log_widget.count() - 1)


class Log(Display, logging.Handler):
    def __init__(self):
        super(Log, self).__init__('Log', size=(800, 100))

        log = QListWidget()

        self.addWidget(log)
        self.log_handler = LogHandler(log)

    def log_exception(self, ex: Exception):
        logger.error('An exception occurred in the experiment. More info to follow:')
        logger.exception(ex)


class Configuration(Display, metaclass=Singleton):
    sigRequestParameters = Signal()
    sigPushParameter = Signal(list, object)

    def __init__(self):
        super(Configuration, self).__init__('Configuration', size=(300, 500))

        container_widget = QWidget()
        layout = QVBoxLayout()

        self.parameter = None
        self.parameter_tree = ParameterTree()
        layout.addWidget(self.parameter_tree)
        container_widget.setLayout(layout)
        self.addWidget(container_widget)

    def request_parameters(self):
        self.sigRequestParameters.emit()

    def update_parameters(self, state: dict):
        self.parameter = GroupParameter(name='top')
        self.parameter.restoreState(state)
        self.parameter_tree.setParameters(self.parameter, showTop=False)  # required to hide top
        self.parameter.sigTreeStateChanged.connect(self.push_changes)

    def push_changes(self, sender, changes: List[Tuple[Parameter, str, Any]]):
        for change in changes:
            if len(change) == 3:
                param, change, info = change
                if change == 'value':
                    self.sigPushParameter.emit(self.parameter.childPath(param), info)


class StateManager(Display, metaclass=Singleton):
    sigStart = Signal()
    sigStop = Signal()
    sigPause = Signal()
    sigReplay = Signal()

    def __init__(self):
        super(StateManager, self).__init__('Status', size=(300, 50))

        self.stop_button = QToolButton()
        self.start_pause_button = QToolButton()
        self.replay_button = QToolButton()
        self.state_label = QLabel('...')

        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.replay_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))

        self.start_pause_button.clicked.connect(self._start_or_pause)
        self.stop_button.clicked.connect(self.sigStop)
        self.replay_button.clicked.connect(self.sigReplay)

        self.start_pause_button.setToolTip('Start/Pause Experiment')
        self.stop_button.setToolTip('Stop Experiment')
        self.replay_button.setToolTip('Replay Experiment')

        layout_widget = QWidget()
        layout_widget.setLayout(QHBoxLayout())

        layout_widget.layout().addWidget(self.stop_button)
        layout_widget.layout().addWidget(self.start_pause_button)
        layout_widget.layout().addWidget(self.replay_button)
        layout_widget.layout().addWidget(self.state_label)

        self.addWidget(layout_widget)

        self.state = CoreState.Connecting

    def update_state(self, state):
        # set state value immediately
        self._state = state
        # defer setter actions until event can be consumed
        invoke_as_event(setattr, self, 'state', state)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state in [CoreState.Starting, CoreState.Pausing, CoreState.Restarting, CoreState.Connecting]:
            self.start_pause_button.setDisabled(True)
            self.stop_button.setDisabled(True)
        elif state in [CoreState.Running]:
            self.start_pause_button.setText('Pause')
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.stop_button.setEnabled(True)
        elif state in [CoreState.Paused]:
            self.start_pause_button.setText('Resume')
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.stop_button.setEnabled(True)
        elif state in [CoreState.Inactive]:
            self.start_pause_button.setText('Start')
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.stop_button.setEnabled(False)

        self.state_label.setText(CoreState(state).name)
        self._state = state

    def _start_or_pause(self):
        if self.start_pause_button.text() == 'Pause':
            self.sigPause.emit()
        elif self.start_pause_button.text() in ['Start', 'Resume']:
            self.sigStart.emit()


class GraphManager(Display, metaclass=Singleton):
    sigPush = Signal(object)

    def __init__(self):
        self.dock_area = DockArea()

        super(GraphManager, self).__init__('Graphs', hideTitle=True, size=(500, 500), widget=self.dock_area)

        self.graphs = dict()  # graph: widget
        graph_signal_relay.sigPush.connect(self.sigPush)

    def set_graphs(self, graphs, data=None):
        self.clear()
        self.graphs.clear()
        for graph in graphs:
            self.register_graph(graph)
        if data:
            self.update_graphs(data, 0)

    def register_graph(self, graph):
        widget = graph.make_widget()
        display = Dock(graph.name, area=self.dock_area, widget=widget)
        # graph.display = display
        self.dock_area.addDock(display, position='below')
        self.graphs[graph] = widget

    def update_graphs(self, data, last_data_size):
        for graph, widget in self.graphs.items():
            try:
                graph.update(widget, data, slice(last_data_size, None))
            except Exception as ex:
                logger.exception(ex)

    def clear(self):
        for graph in self.graphs:
            # NOTE: the parent's parent is always a Display instance containing only that graph
            self.graphs[graph].parent().parent().setParent(None)
            self.graphs[graph].parent().parent().close()
            self.graphs[graph].parent().parent().deleteLater()

    def reset(self):
        self.clear()
        self.set_graphs(self.graphs)
