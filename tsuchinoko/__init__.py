import ctypes
import importlib
import os
import runpy
import sys


import click
from pyqtgraph import mkQApp

from . import parameters  # registers parameter types
from . import patches
from .utils import runengine
try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:
    raise ImportError("You probably haven't installed tsuchinoko yet: pip install -e .") from ex


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


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('path', required=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def bootstrap(path, args):
    """A pyinstaller trick to allow launch of python scripts from built exes"""
    print(path)
    sys.argv.pop(0)
    runpy.run_path(path, {}, "__main__")
