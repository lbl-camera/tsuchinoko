from ._version import get_versions

__version__ = get_versions()['version']

from .utils import runengine

del get_versions

RE = runengine.get_run_engine()
