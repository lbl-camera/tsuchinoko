import pickle
import socket
from queue import Queue
from typing import Callable, Tuple, List, Sequence, Dict
from threading import Thread
import time

import zmq
from loguru import logger
from numpy._typing import ArrayLike

from . import Engine

PORT = 5557
HOST = '127.0.0.1'


class BlueskyAdaptiveEngine(Engine):
    def __init__(self, measure_func=Callable[[Tuple[float]], Tuple[float]]):
        super(BlueskyAdaptiveEngine, self).__init__()

        self.position = None
        self.targets = Queue()
        self.new_measurements = []

        self.setup_socket()

    def setup_socket(self):
        # Setup socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        while True:
            try:
                self.socket.bind(f"tcp://{HOST}:{PORT}")
            except zmq.ZMQError as ex:
                logger.info(f'Unable to bind to tcp://{HOST}:{PORT}. Retrying in 1 second...')
                logger.exception(ex)
                time.sleep(1)
            else:
                logger.info(f'Connected to tcp://{HOST}:{PORT}.')
                break

        # self.zmq_thread = Thread(target=self.communicate)
        # self.zmq_thread.start()

    def update_targets(self, targets: List[Tuple]):
        self.send_payload({'targets': targets})

    def get_measurements(self) -> List[Tuple]:
        new_measurements = []
        # get newly completed measurements from bluesky-adaptive
        while True:
            try:
                payload = self.recv_payload(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            else:
                assert 'target_measured' in payload
                x, y = payload['target_measured']
                new_measurements.append((x,y,1,{}))
        return new_measurements

    def get_position(self) -> Tuple:
        # return last measurement position received from bluesky-adaptive
        return self.position

    def send_payload(self, payload: dict):
        logger.info(f'message: {payload}')
        self.socket.send(pickle.dumps(payload))

    def recv_payload(self, flags=0) -> dict:
        payload_response = pickle.loads(self.socket.recv(flags=flags))
        logger.info(f'response: {payload_response}')
        return payload_response



# ----------------------------------------------------------------------------------------------------------------------


from bluesky_adaptive.agents.base import Agent


class TsuchinokoAgent(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outbound_measurements = []
        self.setup_socket()

    def setup_socket(self):
        # Setup socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        while True:
            try:
                self.socket.connect(f"tcp://{HOST}:{PORT}")
            except zmq.ZMQError:
                logger.info(f'Unable to connect to tcp://{HOST}:{PORT}. Retrying in 1 second...')
                time.sleep(1)
            else:
                logger.info(f'Connected to tcp://{HOST}:{PORT}.')
                break
        self.socket.setsockopt(zmq.CONFLATE, 1)

    def tell(self, x, y):
        payload = {'target_measured': (x, y)}
        self.send_payload(payload)

    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        payload = self.recv_payload()
        assert 'targets' in payload
        return [{}], payload['targets']

    def send_payload(self, payload: dict):
        logger.info(f'message: {payload}')
        self.socket.send(pickle.dumps(payload))

    def recv_payload(self) -> dict:
        payload_response = pickle.loads(self.socket.recv())
        logger.info(f'response: {payload_response}')
        return payload_response

    def measurement_plan(*_):
        pass

    @staticmethod
    def unpack_run(*_):
        pass
