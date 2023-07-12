import pickle
import time
from typing import Tuple, List, Sequence, Dict

import zmq
from loguru import logger
from numpy._typing import ArrayLike

from . import Engine


class BlueskyAdaptiveEngine(Engine):
    def __init__(self, host='127.0.0.1', port=5557):
        super(BlueskyAdaptiveEngine, self).__init__()

        self.position = None
        self.context = None
        self.socket = None
        self.host = host
        self.port = port
        self.setup_socket()

    def setup_socket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)

        # Attempt to bind, retry every second if fails
        while True:
            try:
                self.socket.bind(f"tcp://{self.host}:{self.port}")
            except zmq.ZMQError as ex:
                logger.info(f'Unable to bind to tcp://{self.host}:{self.port}. Retrying in 1 second...')
                logger.exception(ex)
                time.sleep(1)
            else:
                logger.info(f'Bound to tcp://{self.host}:{self.port}.')
                break

    def update_targets(self, targets: List[Tuple]):
        # send targets to TsuchinokoAgent
        self.send_payload({'targets': targets})

    def get_measurements(self) -> List[Tuple]:
        new_measurements = []
        # get newly completed measurements from bluesky-adaptive; repeat until buffered payloads are exhausted
        while True:
            try:
                payload = self.recv_payload(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            else:
                assert 'target_measured' in payload
                x, y = payload['target_measured']
                # TODO: Its highly recommended to extract a variance for y; we might piggyback on y,
                #       s.t. y = [y1, y2, ..., yn, y1variance, y2variance, ..., ynvariance]
                # TODO: Any additional quantities to be interrogated in Tsuchinoko can be included in the trailing dict
                new_measurements.append((x, y, 1, {}))
                # stash the last position measured as the 'current' position of the instrument
                self.position = x
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
# This is a prototype Agent to be used with bluesky-adaptive. This should be extracted before merge.

from bluesky_adaptive.agents.base import Agent


class TsuchinokoAgent(Agent):
    def __init__(self, *args, host='127.0.0.1', port=5557, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port
        self.outbound_measurements = []
        self.setup_socket()

    def setup_socket(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)

        # Attempt to connect, retry every second if fails
        while True:
            try:
                self.socket.connect(f"tcp://{self.host}:{self.port}")
            except zmq.ZMQError:
                logger.info(f'Unable to connect to tcp://{self.host}:{self.port}. Retrying in 1 second...')
                time.sleep(1)
            else:
                logger.info(f'Connected to tcp://{self.host}:{self.port}.')
                break

        # Limit number of buffered messages to 1; dumps any earlier targets if new ones come in before payloads are received
        self.socket.setsockopt(zmq.CONFLATE, 1)

    def tell(self, x, y):
        # Send measurement to BlueskyAdaptiveEngine
        payload = {'target_measured': (x, y)}
        self.send_payload(payload)

    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        # Get targets from BlueskyAdaptiveEngine
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


if __name__ == '__main__':
    # NOTE: change TsuchinokoAgent's base class to `object` to run this primitive mocking of its processes
    agent = TsuchinokoAgent()
    while True:
        _, targets = agent.ask(0)
        agent.tell(targets[0], 1)
