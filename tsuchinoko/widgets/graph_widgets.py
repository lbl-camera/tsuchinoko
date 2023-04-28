import os
import re
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from PySide2.QtCore import Qt, QTimer
from PySide2.QtWidgets import QAction, QFileDialog
from loguru import logger
from pyqtgraph.exporters import ImageExporter
from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from pyqtgraph import InfiniteLine, mkPen, PlotWidget, HistogramLUTWidget, mkBrush, functions as fn, FileDialog

from tsuchinoko.graphics_items.clouditem import CloudItem
from tsuchinoko.graphics_items.indicatoritem import BetterCurveArrow
from tsuchinoko.graphics_items.mixins import ClickRequesterPlot

last_export_directory = None

class CloudWidget(QWidget):
    noRepeatKeys = [
        Qt.Key.Key_Right,
        Qt.Key.Key_Left,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_PageUp,
        Qt.Key.Key_PageDown,
    ]

    def __init__(self, data_key:str, accumulates:bool):
        self.graph = ClickRequesterPlot()
        super().__init__()
        self.data_key = data_key
        self.accumulates = accumulates
        # scatter = ScatterPlotItem(name='scatter', x=[0], y=[0], size=10, pen=mkPen(None), brush=mkBrush(255, 255, 255, 120))
        self.timeline = InfiniteLine(0, pen=mkPen(width=3), movable=True)
        self.timeline.sigPositionChanged.connect(self.timeline_changed)
        self.timeline_plot = PlotWidget()
        self.timeline_plot.hideAxis('left')
        self.timeline_plot.setMouseEnabled(False, False)
        self.timeline_plot.addItem(self.timeline)
        self.timeline_plot.setFixedHeight(40)
        self.cloud = CloudItem(name='scatter', size=10)
        histlut = HistogramLUTWidget()
        histlut.setImageItem(self.cloud)

        self.setLayout(QVBoxLayout())
        hlayout = QHBoxLayout()
        self.layout().addLayout(hlayout)
        self.layout().addWidget(self.timeline_plot)

        hlayout.addWidget(self.graph)
        hlayout.addWidget(histlut)

        self.graph.addItem(self.cloud)

        # Hard-coded to show max
        self.max_arrow = BetterCurveArrow(self.cloud.scatter, brush=mkBrush('r'))
        self.last_arrow = BetterCurveArrow(self.cloud.scatter, brush=mkBrush('w'))
        # text = TextItem()

        self.cache = dict()

        self.keysPressed = {}
        self.play_timer = QTimer()
        self.play_rate = 0
        self._paused_play_rate = None
        self.fps = 1  # 1 Hz by default
        self.last_play_time = 0
        self.play_timer.timeout.connect(self.timeout)

    def timeline_changed(self):
        if not self.cache:
            return
        pos = self.timeline.getXPos()
        self.cloud.updateData(x=self.cache['x'][:int(pos)],
                              y=self.cache['y'][:int(pos)],
                              c=self.cache['v'][:int(pos)],
                              data=self.cache['v'][:int(pos)],
                              hoverable=True,
                              hoverPen=mkPen('b', width=2))

    def nframes(self):
        if self.cache:
            return len(self.cache['v'])

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.toggle_pause()
            ev.accept()
        elif ev.key() == Qt.Key.Key_Home:
            self.set_current_index(0)
            self.play(0)
            ev.accept()
        elif ev.key() == Qt.Key.Key_End:
            self.set_current_index(self.nframes()-1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.eval_key_state()
        else:
            super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() in [Qt.Key.Key_Space, Qt.Key.Key_Home, Qt.Key.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.eval_key_state()
        else:
            super().keyReleaseEvent(ev)

    def eval_key_state(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == Qt.Key.Key_Right:
                self.play(20)
                self.jump_frames(1)
                # effectively pause playback for 0.2 s
                self.last_play_time = perf_counter() + 0.2
            elif key == Qt.Key.Key_Left:
                self.play(-20)
                self.jump_frames(-1)
                self.last_play_time = perf_counter() + 0.2
            elif key == Qt.Key.Key_Up:
                self.play(-100)
            elif key == Qt.Key.Key_Down:
                self.play(100)
            elif key == Qt.Key.Key_PageUp:
                self.play(-1000)
            elif key == Qt.Key.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)

    def timeout(self):
        now = perf_counter()
        dt = now - self.last_play_time
        if dt < 0:
            return
        n = int(self.play_rate * dt)
        if n != 0:
            self.last_play_time += (float(n) / self.play_rate)
            if self.timeline.getXPos() + n >= self.nframes():
                self.play(0)
            self.jump_frames(n)

    def play(self, rate=None):
        """Begin automatically stepping frames forward at the given rate (in fps).
        This can also be accessed by pressing the spacebar."""
        if rate is None:
            rate = self._paused_play_rate or self.fps
        if rate == 0 and self.play_rate not in (None, 0):
            self._paused_play_rate = self.play_rate
        self.play_rate = rate

        if rate == 0:
            self.play_timer.stop()
            return

        self.last_play_time = perf_counter()
        if not self.play_timer.isActive():
            self.play_timer.start(abs(int(1000 / rate)))

    def toggle_pause(self):
        if self.play_timer.isActive():
            self.play(0)
        elif self.play_rate == 0:
            if self._paused_play_rate is not None:
                fps = self._paused_play_rate
            else:
                fps = (self.nframes() - 1) / (self.tVals[-1] - self.tVals[0])
            self.play(fps)
        else:
            self.play(self.play_rate)

    def set_current_index(self, ind):
        """Set the currently displayed frame index."""
        index = int(fn.clip_scalar(ind, 0, self.nframes()-1))
        self.timeline.setValue(index)

    def jump_frames(self, n):
        """Move video frame ahead n frames (may be negative)"""
        self.set_current_index(self.timeline.getXPos() + n)

    def update_data(self, data, update_slice: slice):
        # require_clear = False
        timeline_at_end = not self.cache or self.timeline.getXPos() == len(self.cache['v']) - 1

        with data.r_lock():
            v = data[self.data_key].copy()

            x, y = zip(*data.positions)

        lengths = len(v), len(x), len(y)
        min_length = min(lengths)

        if not np.all(np.array(lengths) == min_length):
            logger.warning(f'Ragged arrays passed to cloud item with lengths (v, x, y): {lengths}')
            x = x[:min_length]
            y = y[:min_length]
            v = v[:min_length]

        if not len(x):
            return

        self.cache = {'x': x, 'y': y, 'v': v}

        max_index = np.argmax(v)
        last_data_size = min(update_slice.start, len(self.cloud.cData))

        if last_data_size == 0 and timeline_at_end:
            action = self.cloud.setData
        elif not self.accumulates:
            action = self.cloud.updateData
        elif timeline_at_end:
            action = self.cloud.extendData
            x = x[last_data_size + 1:]
            y = y[last_data_size + 1:]
            v = v[last_data_size + 1:]
        else:
            action = None

        if action:
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

        self.max_arrow.setIndex(max_index)
        self.last_arrow.setIndex(len(self.cloud.cData) - 1)

        self.timeline_plot.setXRange(0, min_length-1, padding=0.01)
        self.timeline.setBounds([0, min_length-1])
        if timeline_at_end:
            with fn.SignalBlock(self.timeline.sigPositionChanged, self.timeline_changed):
                self.timeline.setPos(min_length-1)

        # text.setText(f'Max: {v[max_index]:.2f} ({x[max_index]:.2f}, {y[max_index]:.2f})')
        # text.setPos(x[max_index], y[max_index])

    def getView(self):
        return self.graph