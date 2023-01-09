from dataclasses import dataclass
from enum import Enum, auto

from qtpy.QtWidgets import QWidget


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


@dataclass(eq=False)
class Graph:
    name: str
    compute_with: Location = Location.Client
    compute_mode: ComputeMode = ComputeMode.Blocking
    render_mode: RenderMode = RenderMode.Blocking
    widget: QWidget = None

    def compute(self, data: 'Data', *args):
        ...

    def update(self, data: 'Data', *args):
        ...

    def make_widget(self):
        ...

    # @property
    # def widget(self):
    #     return
    #
    # @widget.setter
    # def widget(self, value):
    #     self._widget = value
