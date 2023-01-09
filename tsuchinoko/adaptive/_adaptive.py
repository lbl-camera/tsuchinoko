# This module is separated to limit imports done in child processes
from multiprocessing import Queue
import time


targets = Queue()
measurements = Queue()


def measure(position):
    targets.put(position)
    return measurements.get()


def set_global(*args):
    global targets, measurements
    targets, measurements = args
    # print('ID:', id(targets))
    # print(targets, measurements)
