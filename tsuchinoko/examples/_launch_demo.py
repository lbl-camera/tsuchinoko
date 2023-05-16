"""
Launches a client window; attempts to connect at localhost address
"""
import tsuchinoko
import sys
import faulthandler

if __name__ == '__main__':
    try:
        faulthandler.enable()
    except AttributeError:
        pass

    tsuchinoko.launch_server(sys.argv[1:])
