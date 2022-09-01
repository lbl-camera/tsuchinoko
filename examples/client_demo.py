"""
Launches a client window; attempts to connect at localhost address
"""

from pyqtgraph import mkQApp

from tsuchinoko.widgets.mainwindow import MainWindow

if __name__ == '__main__':
    qapp = mkQApp('Tsuchinoko')

    main_window = MainWindow()
    main_window.show()

    exit(qapp.exec_())
