from pyqtgraph import mkQApp
from qtpy.QtWidgets import QApplication

from tsuchinoko.widgets.mainwindow import MainWindow

if __name__ == '__main__':
    import pyqtgraph as pg

    qapp = mkQApp('Tsuchinoko')

    main_window = MainWindow()

    # main_window.experiment = AlignmentExperiment(main_window.graph_manager_widget)

    main_window.show()

    # iv = pg.PlotWidget()
    # scatter = pg.ScatterPlotItem(x=[0], y=[0], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
    # arrow = pg.CurveArrow(scatter)
    # text = pg.TextItem()
    # iv.addItem(scatter)
    # iv.addItem(arrow)
    # iv.addItem(text)

    exit(qapp.exec_())
