"""
Launches a client window; attempts to connect at localhost address
"""
import tsuchinoko
import sys

if __name__ == '__main__':
    tsuchinoko.launch_server(sys.argv[1:])
