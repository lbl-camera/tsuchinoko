import sys


class StdOutPatch:
    def flush(self):
        pass

    def write(self, *args, **kwargs):
        pass


# Jupyter kernels try to flush stdout/stderr, but these might not exist in all environments
for channel in ['stdout', 'stderr']:
    if getattr(sys, channel) is None:
        setattr(sys, channel, StdOutPatch())
