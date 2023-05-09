"""
Launches a client window; attempts to connect at localhost address
"""
import tsuchinoko
import sys

if __name__ == '__main__':
    tsuchinoko.bootstrap(sys.argv[1:])
