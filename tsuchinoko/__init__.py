import importlib
import os
import runpy
import sys


import click

try:
    from ._version import __version__
except (ImportError, ModuleNotFoundError) as ex:
    raise ImportError("You probably haven't installed tsuchinoko yet: pip install -e .") from ex

@click.command()
@click.argument('core_address', required=False, default='localhost')
def launch_client(core_address='localhost'):
    """Launch the Qt GUI client (requires tsuchinoko[gui])."""
    try:
        from pyqtgraph import mkQApp
        from . import parameters, patches
    except ImportError:
        raise click.ClickException("GUI requires PySide6: pip install tsuchinoko[gui]")

    import ctypes
    if os.name == 'nt':
        myappid = 'camera.tsuchinoko'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

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
