import pickle
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Sequence, Dict

import zmq
from loguru import logger
from numpy._typing import ArrayLike

from . import Engine

SLEEP_FOR_AGENT_TIME = .1
SLEEP_FOR_TSUCHINOKO_TIME = .1
FORCE_KICKSTART_TIME = 5


class BlueskyAdaptiveEngine(Engine):
    """
    A `tsuchinoko.adaptive.Engine` that sends targets to Blueskly-Adaptive and receives back measured data.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 5557):
        """

        Parameters
        ----------
        host
            A host address target for the zmq socket.
        port
            The port used for the zmq socket.
        """
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
                x, (y, v) = payload['target_measured']
                # TODO: Any additional quantities to be interrogated in Tsuchinoko can be included in the trailing dict
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


# ----------------------------------------------------------------------------------------------------------------------
# This is a prototype Agent to be used with bluesky-adaptive. This should be extracted before merge.

from bluesky_adaptive.agents.base import Agent


class TsuchinokoBase(ABC):
    def __init__(self, *args, host: str = '127.0.0.1', port: int = 5557, **kwargs):
        """

        Parameters
        ----------
        args
            args passed through to `bluesky_adaptive.agents.base.Agent.__init__()`
        host
            A host address target for the zmq socket.
        port
            The port used for the zmq socket.
        kwargs
            kwargs passed through to `bluesky_adaptive.agents.base.Agent.__init__()`
        """

        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port
        self.outbound_measurements = []
        self.context = None
        self.socket = None
        self.setup_socket()
        self.last_targets_received = time.time()
        self.kickstart()

    def kickstart(self):
        self.send_payload({'send_targets': True})  # kickstart to recover from shutdowns
        self.last_targets_received = time.time()  # forgive lack of response until now

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

    def tell(self, x, y, v):
        """
        Send measurement to BlueskyAdaptiveEngine
        """
        yv = (y, v)
        payload = {'target_measured': (x, yv)}
        self.send_payload(payload)

    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        """
        Wait until at least one target is received, also exhaust the queue of received targets, overwriting old ones
        """
        payload = None
        while True:
            try:
                payload = self.recv_payload(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                if payload is not None:
                    break
                else:
                    time.sleep(SLEEP_FOR_TSUCHINOKO_TIME)
                    if time.time() > self.last_targets_received + FORCE_KICKSTART_TIME:
                        self.kickstart()
        assert 'targets' in payload
        self.last_targets_received = time.time()
        return payload['targets']

    def send_payload(self, payload: dict):
        logger.info(f'message: {payload}')
        self.socket.send(pickle.dumps(payload))

    def recv_payload(self, flags=0) -> dict:
        payload_response = pickle.loads(self.socket.recv(flags=flags))
        logger.info(f'response: {payload_response}')
        return payload_response


class TsuchinokoAgent(TsuchinokoBase, Agent):
    """
    A Bluesky-Adaptive 'Agent'. This Agent communicates with Tsuchinoko over zmq to request new targets and report back
    measurements. This is an abstract class that must be subclassed.

    A `tsuchinoko.execution.bluesky_adaptive.BlueskyAdaptiveEngine` is required for the Tsuchinoko server to complement
    one of these `TsuchinokoAgent`.
    """

    def tell(self, x, y, v) -> Dict[str, ArrayLike]:
        super().tell(x, y, v)
        return self.get_tell_document(x, y, v)

    def ask(self, batch_size: int) -> Tuple[Sequence[Dict[str, ArrayLike]], Sequence[ArrayLike]]:
        targets = super().ask(batch_size)
        return self.get_ask_documents(targets), targets

    @abstractmethod
    def get_tell_document(self, x, y, v) -> Dict[str, ArrayLike]:
        """
        Return any single document corresponding to 'tell'-ing Tsuchinoko about the newly measured `x`, `y` data

        Parameters
        ----------
        x :
            Independent variable for data observed
        y :
            Dependent variable for data observed
        v :
            Variance for measurement of y

        Returns
        -------
        dict
            Dictionary to be unpacked or added to a document

        """
        ...

    @abstractmethod
    def get_ask_documents(self, targets: List[Tuple]) -> Sequence[Dict[str, ArrayLike]]:
        """
        Ask the agent for a new batch of points to measure.

        Parameters
        ----------
        targets : List[Tuple]
            The new target positions to be measured received during this `ask`.

        Returns
        -------
        docs : Sequence[dict]
            Documents of key metadata from the ask approach for each point in next_points.
            Must be length of batch size.

        """
        ...


if __name__ == '__main__':
    # NOTE: This usage is a primitive mocking of Bluesky-Adaptive's processes
    agent = TsuchinokoBase()
    while True:
        targets = agent.ask(0)
        agent.tell(targets[0], 1)
