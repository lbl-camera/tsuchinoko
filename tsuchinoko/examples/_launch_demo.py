"""
Launches a client window; attempts to connect at localhost address
"""
import tsuchinoko
import sys
import faulthandler

if __name__ == '__main__':
    faulthandler.enable()
    tsuchinoko.launch_server(sys.argv[1:])
