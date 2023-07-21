import pickle
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Sequence, Dict, Union

import zmq
from databroker.client import BlueskyRun
from loguru import logger
from numpy._typing import ArrayLike

from . import Engine

SLEEP_FOR_AGENT_TIME = .1
SLEEP_FOR_TSUCHINOKO_TIME = .1


class BlueskyAdaptiveEngine(Engine):
    def __init__(self, host='127.0.0.1', port=5557):
        super(BlueskyAdaptiveEngine, self).__init__()

        self.position = None
        self.context = None
        self.socket = None
        self.host = host
        self.port = port
        self.setup_socket()
        self._last_targets_sent = None
        # Lock sending new points until at least one from the previous list is measured
        self.has_fresh_points_on_server = False

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
        if self.has_fresh_points_on_server:
            time.sleep(SLEEP_FOR_AGENT_TIME)  # chill if the Agent hasn't measured any points from the previous list
        else:
            # send targets to TsuchinokoAgent
            self.has_fresh_points_on_server = self.send_payload({'targets': targets})
            self._last_targets_sent = targets

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
                new_measurements.append((x, y, [1]*len(y), {}))
                # stash the last position measured as the 'current' position of the instrument
                self.position = x
        if new_measurements:
            self.has_fresh_points_on_server = False
        return new_measurements

    def get_position(self) -> Tuple:
        # return last measurement position received from bluesky-adaptive
        return self.position

    def send_payload(self, payload: dict):
        logger.info(f'message: {payload}')
        try:
            self.socket.send(pickle.dumps(payload), flags=zmq.NOBLOCK)
        except zmq.error.Again:
            return False
        return True

    def recv_payload(self, flags=0) -> dict:
        payload_response = pickle.loads(self.socket.recv(flags=flags))
        logger.info(f'response: {payload_response}')
        # if the returned message is the kickstart message, resend the last targets sent and check for more payloads
        if payload_response == {'send_targets': True}:
            self.has_fresh_points_on_server = False
            self.update_targets(self._last_targets_sent)
            payload_response = self.recv_payload(flags)
        return payload_response


# ----------------------------------------------------------------------------------------------------------------------
# This is a prototype Agent to be used with bluesky-adaptive. This should be extracted before merge.

from bluesky_adaptive.agents.base import Agent


class TsuchinokoBase(ABC):
    def __init__(self, *args, host='127.0.0.1', port=5557, **kwargs):
        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port
        self.outbound_measurements = []
        self.context = None
        self.socket = None
        self.setup_socket()
        self.send_payload({'send_targets': True})  # kickstart to recover from shutdowns

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

    def tell(self, x, y) -> Dict[str, ArrayLike]:
        # Send measurement to BlueskyAdaptiveEngine
        payload = {'target_measured': (x, y)}
        self.send_payload(payload)
        return {}

    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        # Wait until at least one target is received, also exhaust the queue of received targets, overwriting old ones
        payload = None
        while True:
            try:
                payload = self.recv_payload(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                if payload is not None:
                    break
                else:
                    time.sleep(SLEEP_FOR_TSUCHINOKO_TIME)
        assert 'targets' in payload
        return [{}], payload['targets']

    def send_payload(self, payload: dict):
        logger.info(f'message: {payload}')
        self.socket.send(pickle.dumps(payload))

    def recv_payload(self, flags=0) -> dict:
        payload_response = pickle.loads(self.socket.recv(flags=flags))
        logger.info(f'response: {payload_response}')
        return payload_response


class TsuchinokoAgent(TsuchinokoBase, ABC):

    @abstractmethod
    def measurement_plan(self, point: ArrayLike) -> Tuple[str, List, dict]:
        ...

    @staticmethod
    @abstractmethod
    def unpack_run(run: BlueskyRun) -> Tuple[Union[float, ArrayLike], Union[float, ArrayLike]]:
        ...


if __name__ == '__main__':
    # NOTE: This usage is a primitive mocking of Bluesky-Adaptive's processes
    agent = TsuchinokoBase()
    while True:
        _, targets = agent.ask(0)
        agent.tell(targets[0], 1)
