from loguru import logger

from ..execution import Engine as ExecutionEngine
from ..adaptive import Engine as AdaptiveEngine, Data
from ..utils.logging import log_time


class Core:
    def __init__(self):
        self.execution_engine: ExecutionEngine = None
        self.adaptive_engine: AdaptiveEngine = None

        self.iteration = 0

    def set_execution_engine(self, engine: ExecutionEngine):
        self.execution_engine = engine

    def set_adaptive_engine(self, engine: AdaptiveEngine):
        self.adaptive_engine = engine

    async def start(self):
        data = Data(dimensionality=self.adaptive_engine.dimensionality)

        while True:
            logger.info(f'Iteration: {len(data)}')
            with log_time('getting position', cumulative_key='getting position'):
                position = tuple(self.execution_engine.get_position())
            with log_time('getting targets', cumulative_key='getting targets'):
                targets = self.adaptive_engine.request_targets(position, 10)
            with log_time('updating targets', cumulative_key='updating targets'):
                self.execution_engine.update_targets(targets)
            with log_time('getting measurements', cumulative_key='getting measurements'):
                new_measurements = self.execution_engine.get_measurements()
            if len(new_measurements):
                with log_time('injecting new measurements', cumulative_key='injecting new measurements'):
                    data.inject_new(new_measurements)
                    self.adaptive_engine.update_measurements(data)
            with log_time('informing clients', cumulative_key='informing clients'):
                await self.notify_clients(data)

            if not (len(data) % 2000) and len(data):
                with log_time('training', cumulative_key='training'):
                    self.adaptive_engine.train()

    async def notify_clients(self, data):
        ...


class ZMQCore(Core):
    def __init__(self):
        super(ZMQCore, self).__init__()

    def start_server(self):
        import zmq
        from zmq.asyncio import Context
        self.context = Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

    async def notify_clients(self, data: Data):
        import zmq
        if self.socket:
            try:
                message = await self.socket.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            else:
                logger.info("Received request: %s" % message)
                import json
                if message == b'full_data':
                    await self.socket.send_string(json.dumps(data.as_dict()))
                elif b'partial_data' in message:
                    start = int(message.split(b' ')[1])
                    partial_data = data[start:]
                    await self.socket.send_string(json.dumps(partial_data.as_dict()))

