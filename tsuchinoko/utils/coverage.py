# Code coverage tools 'coverage' and 'pytest-cov' don't seem to correctly trace
# code which is inside methods called from within QThreads, see
# https://github.com/nedbat/coveragepy/issues/686
# To mitigate this problem, I use a custom decorator '@coverage_resolve_trace'
# to be hung onto those method definitions. This will prepend the decorated
# method code with 'sys.settrace(threading._trace_hook)' when a code
# coverage test is detected. When no coverage test is detected, it will just
# pass the original method untouched.
import sys
import threading
from functools import wraps

running_coverage = 'coverage' in sys.modules
if running_coverage: print("\nCode coverage test detected\n")


def coverage_resolve_trace(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if running_coverage: sys.settrace(threading._trace_hook)
        fn(*args, **kwargs)

    return wrapped
