from ._version import get_versions

__version__ = get_versions()['version']

from .utils import runengine
from . import parameters  # registers parameter types

del get_versions

RE = runengine.get_run_engine()
