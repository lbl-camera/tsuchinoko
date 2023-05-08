import click
from qtpy.QtCore import QEventLoop
from pyqtgraph import mkQApp
import ctypes
import os
import asyncio
import importlib
import sys

from ._version import get_versions

__version__ = get_versions()['version']

from . import patches
from .utils import runengine
from . import parameters  # registers parameter types

del get_versions


@click.command()
@click.argument('core_address', required=False, default='localhost')
def launch_client(core_address='localhost'):

    if os.name == 'nt':
        # https://stackoverflow.com/questions/67599432/setting-the-same-icon-as-application-icon-in-task-bar-for-pyqt5-application
        myappid = 'camera.tsuchinoko'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)  # Allows taskbar icon to be shown on windows

    from .widgets.mainwindow import MainWindow
    qapp = mkQApp('Tsuchinoko')

    main_window = MainWindow(core_address)
    main_window.show()

    sys.exit(qapp.exec_())


@click.command()
@click.argument('demo_name', required=False, default='server_demo')
def launch_server(demo_name='server_demo'):
    demo_module = importlib.import_module(f'tsuchinoko.examples.{demo_name}')
    demo_module.core.main()


@click.command()
@click.argument('path', required=True)
def bootstrap(path):
    """A pyinstaller trick to allow launch of python scripts from built exes"""
    print(path)
    exec(f"""__name__='__main__'; {open(path).read()}""")

