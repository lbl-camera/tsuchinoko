import numbers
import pickle
import time
from typing import Tuple, List

import numpy as np
import zmq
from loguru import logger

from . import Engine
from ..adaptive.gpCAM_in_process import GPCAMInProcessEngine

SLEEP_FOR_AGENT_TIME = .1
SLEEP_FOR_TSUCHINOKO_TIME = .1
FORCE_KICKSTART_TIME = 5


class BlueskyAdaptiveEngine(Engine):
    """
    A `tsuchinoko.execution.Engine` that sends targets to Blueskly-Adaptive and receives back measured data.
    """

    suggest_blacklist = ["x_data",
                         "y_data",
                         "noise_variances",
                         "init_hyperparameters"]  # keys with ragged state

    def __init__(self, adaptive_engine:GPCAMInProcessEngine, host: str = '127.0.0.1', port: int = 5557):
        """

        Parameters
        ----------
        host
            A host address target for the zmq socket.
        port
            The port used for the zmq socket.
        """
        super(BlueskyAdaptiveEngine, self).__init__()

        self.adaptive_engine = adaptive_engine
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
            # checkpoint optimizer state
            gpcam_state = self.adaptive_engine.optimizer.__getstate__()

            # sanitize state
            for key in self.suggest_blacklist:
                if gpcam_state[key] is None:
                    gpcam_state[key] = np.array([])
            sanitized_gpcam_state = dict(
                **{key if key not in self.suggest_blacklist else f"STATEDICT-{key}":
                       np.asarray(val) if isinstance(val, (list, tuple)) else val
                   for key, val in gpcam_state.items()
                   if val is not None}  # event-model doesn't like Nones
                )

            # send targets to TsuchinokoAgent
            self.has_fresh_points_on_server = self.send_payload({'candidate': targets,
                                                                 'optimizer': sanitized_gpcam_state})
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
                x, (y, v) = payload['target_measured']
                new_measurements.append((x, y, v, {}))
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
