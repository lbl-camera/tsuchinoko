import numpy as np
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication, QWidget, QHBoxLayout, QSlider
from ophyd.sim import SynAxis, SynSignal
from scipy.stats import multivariate_normal

from tsuchinoko.main import alignment_plan
from tsuchinoko.utils import runengine

if __name__ == '__main__':
    import pyqtgraph as pg

    min1 = -3
    max1 = 3
    min2 = -3
    max2 = 3
    pinhole1min = -3
    pinhole1max = 3
    pinhole2min = -3
    pinhole2max = 3
    pinhole_min_step = .01

    motor1 = SynAxis(name='motor1', labels={'motors'})
    motor2 = SynAxis(name='motor2', labels={'motors'})
    pinhole1 = SynAxis(name='pinhole1', labels={'motors'})
    pinhole2 = SynAxis(name='pinhole2', labels={'motors'})


    def measure_monitor():
        beam_center = motor1.readback.get(), motor2.readback.get()
        beam_stddev = motor1.readback.get() ** 2 + .5, motor2.readback.get() ** 2 + .5
        pinhole = pinhole1.readback.get(), pinhole2.readback.get()
        # print(beam_center, beam_stddev, pinhole)

        return multivariate_normal.pdf(pinhole,
                                       beam_center,
                                       np.diag(np.asarray(beam_stddev) ** 2))  # * np.random.rand()*1e-1


    monitor = SynSignal(name='monitor',
                        labels={'monitor'},
                        func=measure_monitor)

    qapp = QApplication([])

    w = QWidget()
    w.setLayout(QHBoxLayout())
    iv = pg.PlotWidget()
    scatter = pg.ScatterPlotItem(x=[0], y=[0], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    arrow = pg.CurveArrow(scatter)
    text = pg.TextItem()
    slider = QSlider(orientation=Qt.Vertical)
    slider.setMinimum(-100)
    slider.setMaximum(100)

    w.layout().addWidget(iv)
    w.layout().addWidget(slider)

    iv.addItem(scatter)
    iv.addItem(arrow)
    iv.addItem(text)
    w.show()
    N = 1000

    RE = runengine.get_run_engine()

    plan = alignment_plan(monitor, motor1, min1, max1, motor2, min2, max2, pinhole1, pinhole1min, pinhole1max, pinhole2,
                          pinhole2min, pinhole2max, pinhole_min_step, 10000)
    RE(plan)

    qapp.exec_()
