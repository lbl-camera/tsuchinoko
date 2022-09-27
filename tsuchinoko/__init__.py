import click
from pyqtgraph import mkQApp
import ctypes
import os

from ._version import get_versions

__version__ = get_versions()['version']

from .utils import runengine
from . import parameters  # registers parameter types


del get_versions

RE = runengine.get_run_engine()


@click.command()
@click.argument('core_address', required=False, default='localhost')
def launch_client(core_address='localhost'):

    if os.name == 'nt':
        # https://stackoverflow.com/questions/67599432/setting-the-same-icon-as-application-icon-in-task-bar-for-pyqt5-application
        myappid = 'camera.xicam'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)  # Allows taskbar icon to be shown on windows

    from .widgets.mainwindow import MainWindow
    qapp = mkQApp('Tsuchinoko')

    main_window = MainWindow(core_address)
    main_window.show()

    exit(qapp.exec_())

