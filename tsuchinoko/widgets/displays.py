import logging

from PySide2.QtCore import QObject, Signal
from PySide2.QtGui import QBrush, Qt
from pyqtgraph.dockarea import Dock, DockArea
from qtpy.QtWidgets import QDoubleSpinBox, QCheckBox, QFormLayout, QWidget, QListWidget, QListWidgetItem, QPushButton, QLabel, QSpacerItem, QSizePolicy, QStyle, QToolButton, QHBoxLayout

from tsuchinoko import RE
from tsuchinoko.core import CoreState


class Singleton(type(QObject)):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Display(Dock):
    ...


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


class LogHandler(logging.Handler):
    colors = {logging.DEBUG: Qt.gray, logging.ERROR: Qt.darkRed, logging.CRITICAL: Qt.red,
              logging.INFO: Qt.white, logging.WARNING: Qt.yellow}

    def __init__(self, log_widget, level=logging.DEBUG):
        super(LogHandler, self).__init__(level=level)
        logging.getLogger().addHandler(self)
        self.log_widget = log_widget

    def emit(self, record, level=logging.INFO, timestamp=None, icon=None, *args):  # We can have icons!
        item = QListWidgetItem(record.getMessage())
        item.setForeground(QBrush(self.colors[record.levelno]))
        item.setToolTip(timestamp)
        self.log_widget.insertItem(0, item)

        while self.log_widget.count() > 100:
            self.log_widget.takeItem(self.log_widget.count() - 1)


class Log(Display, logging.Handler):
    def __init__(self):
        super(Log, self).__init__('Log', size=(800, 300))

        log = QListWidget()

        self.addWidget(log)
        self.log_handler = LogHandler(log)


class Configuration(Display, metaclass=Singleton):
    def __init__(self):
        super(Configuration, self).__init__('Configuration', size=(300, 500))

        container_widget = QWidget()

        mean_weight_default, stddev_x_weight_default, stddev_y_weight_default = [1e0, 1e-1, 1e-1]
        posterior_weight_factor_default = 3

        self.posterior_weight_factor = QDoubleSpinBox()
        self.posterior_weight_factor.setValue(posterior_weight_factor_default)
        self.mean_weight = QDoubleSpinBox()
        self.mean_weight.setValue(mean_weight_default)
        self.stddev_x_weight = QDoubleSpinBox()
        self.stddev_x_weight.setValue(stddev_x_weight_default)
        self.stddev_y_weight = QDoubleSpinBox()
        self.stddev_y_weight.setValue(stddev_y_weight_default)
        self.debug_fit = QCheckBox()

        form_layout = QFormLayout()
        form_layout.addRow('Post. Mean/Covariance Weighting', self.posterior_weight_factor)
        form_layout.addRow('Amplitude weight', self.mean_weight)
        form_layout.addRow('Std. Dev. X weight', self.stddev_x_weight)
        form_layout.addRow('Std. Dev. Y weight', self.stddev_y_weight)
        form_layout.addRow('Debug centroid/tune/fit', self.debug_fit)

        container_widget.setLayout(form_layout)
        self.addWidget(container_widget)


class RunEngineControls(Display, metaclass=Singleton):
    def __init__(self):
        super(RunEngineControls, self).__init__('Controls')

        self.experiment = None

        container_widget = QWidget()
        self.start = QPushButton('Start')
        self.pause = QPushButton('Pause')
        self.resume = QPushButton('Resume')
        self.pause.hide()
        self.resume.hide()
        self.measurement_time = QLabel('...')
        self.cycle_time = QLabel('...')

        form_layout = QFormLayout()
        form_layout.addWidget(self.start)
        form_layout.addWidget(self.pause)
        form_layout.addWidget(self.resume)
        form_layout.addItem(QSpacerItem(1, 1, vData=QSizePolicy.Expanding))
        form_layout.addRow('Measurement Time:', self.measurement_time)
        form_layout.addRow('Cycle Time:', self.cycle_time)

        container_widget.setLayout(form_layout)
        self.addWidget(container_widget)

        self.start.clicked.connect(self.start_plan)
        self.pause.clicked.connect(self.pause_plan)
        self.resume.clicked.connect(self.resume_plan)

    def start_plan(self):
        RE(self.experiment.plan)
        self.start.hide()
        self.pause.show()

    def pause_plan(self):
        RE.pause()
        self.resume.show()
        self.pause.hide()

    def resume_plan(self):
        RE.resume()
        self.resume.hide()
        self.pause.show()


class StateManager(Display, metaclass=Singleton):
    sigStart = Signal()
    sigStop = Signal()
    sigPause = Signal()

    def __init__(self):
        super(StateManager, self).__init__('Tsuchinoko Status', size=(300, 100))

        self.state = CoreState.Connecting

        self.stop_button = QToolButton()
        self.start_pause_button = QToolButton()
        self.state_label = QLabel('...')

        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.start_pause_button.clicked.connect(self._start_or_pause)
        self.stop_button.clicked.connect(self.sigStop)

        layout_widget = QWidget()
        layout_widget.setLayout(QHBoxLayout())

        layout_widget.layout().addWidget(self.stop_button)
        layout_widget.layout().addWidget(self.start_pause_button)
        layout_widget.layout().addWidget(self.state_label)

        self.addWidget(layout_widget)

        self.update_state(self.state)

    def update_state(self, state):
        if state in [CoreState.Starting, CoreState.Pausing, CoreState.Restarting, CoreState.Connecting]:
            self.start_pause_button.setDisabled(True)
            self.stop_button.setDisabled(True)
        elif state in [CoreState.Running]:
            self.start_pause_button.setText('Pause')
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        elif state in [CoreState.Paused]:
            self.start_pause_button.setText('Resume')
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        elif state in [CoreState.Inactive]:
            self.start_pause_button.setText('Start')
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.state_label.setText(CoreState(state).name)
        self.state = state

    def _start_or_pause(self):
        if self.start_pause_button.text() == 'Pause':
            self.sigPause.emit()
        elif self.start_pause_button.text() in ['Start', 'Resume']:
            self.sigStart.emit()


class GraphManager(Display, metaclass=Singleton):
    def __init__(self):
        super(GraphManager, self).__init__('Graphs', hideTitle=True, size=(500, 500))
        self.dock_area = DockArea()
        self.addWidget(self.dock_area)

        self.graphs = dict()
        self.update_callbacks = dict()

    def register_graph(self, name, widget, update_callback):
        display = Display(name)
        display.addWidget(widget)
        self.dock_area.addDock(display, 'below')

        self.graphs[name] = widget
        self.update_callbacks[name] = update_callback

    def update(self, data, last_data_size):
        for update_callback in self.update_callbacks.values():
            update_callback(data, last_data_size)
