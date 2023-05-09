from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar
from uuid import uuid4

from qtpy.QtWidgets import QWidget
from qtpy.QtCore import Signal, QObject


class Location(Enum):
    Client = auto()
    Core = auto()
    ExecutionEngine = auto()
    AdaptiveEngine = auto()


class ComputeMode(Enum):
    Blocking = auto()
    Threaded = auto()


class RenderMode(Enum):
    Blocking = auto()
    Background = auto()


class GraphSignalRelay(QObject):
    sigPush = Signal(object)


graph_signal_relay = GraphSignalRelay()


@dataclass(eq=False)
class Graph:
    name: ClassVar[str] = ''
    compute_with: Location = Location.Client
    compute_mode: ComputeMode = ComputeMode.Blocking
    render_mode: RenderMode = RenderMode.Blocking
    widget_class: ClassVar[type[QWidget]] = None
    widget_args: tuple = field(default_factory=tuple)
    widget_kwargs: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


    def compute(self, data: 'Data', *args):
        ...

    def update(self, data: 'Data', *args):
        ...

    def make_widget(self):
        return self.widget_class(*self.widget_args, **self.widget_kwargs)

    # @property
    # def widget(self):
    #     return
    #
    # @widget.setter
    # def widget(self, value):
    #     self._widget = value
