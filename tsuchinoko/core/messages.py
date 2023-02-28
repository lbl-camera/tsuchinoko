import enum


class _SimpleReprEnum(enum.Enum):
    def __repr__(self):
        return self.name


class Role(_SimpleReprEnum):
    CLIENT = enum.auto()
    SERVER = enum.auto()


class Direction(_SimpleReprEnum):
    RESPONSE = enum.auto()
    REQUEST = enum.auto()


Commands = {}
Commands[Role.CLIENT] = {}
Commands[Role.SERVER] = {}
_commands = set()

ENCODING = 'ascii'


class _MetaDirectionalMessage(type):
    # see what you've done, @tacaswell and @danielballan?
    def __new__(metacls, name, bases, dct):
        if name.endswith('Request'):
            direction = Direction.REQUEST
            command_dict = Commands[Role.CLIENT]
        else:
            direction = Direction.RESPONSE
            command_dict = Commands[Role.SERVER]

        dct['DIRECTION'] = direction
        new_class = super().__new__(metacls, name, bases, dct)

        if new_class.FNC is not None:
            command_dict[new_class.FNC] = new_class

        if not name.startswith('_'):
            _commands.add(new_class)
        return new_class


class Message(metaclass=_MetaDirectionalMessage):
    __slots__ = ('payload',)
    WRITE_REQUIRED = False
    FNC = None

    def __init__(self, *payload):
        self.payload = payload
        for name, value in zip(self.__slots__, payload):
            setattr(self, name, value)

    # @classmethod
    # def from_wire(cls, payload_buffers):
    #     return cls.from_components(
    #         str(b''.join(payload_buffers), ENCODING).strip())

    # @classmethod
    # def from_components(cls, str_payload):
    #     # Bwahahahaha
    #     instance = cls.__new__(cls)
    #     instance.str_payload = str_payload
    #     return instance

    # def __bytes__(self):
    #     return bytes(self.str_payload, ENCODING)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.payload!r}'


class _DataResponse(Message):
    def __repr__(self):
        return f'{self.__class__.__name__} size: {len(self.payload[0]["positions"])}'


class PushDataRequest(Message):
    __slots__ = ('data',)


class PushDataResponse(Message):
    __slots__ = ()


class FullDataRequest(Message):
    __slots__ = ()


class FullDataResponse(_DataResponse):
    __slots__ = ('data',)


class PartialDataRequest(Message):
    __slots__ = ('iteration',)


class PartialDataResponse(_DataResponse):
    __slots__ = ('data', 'last_data_size')


class StartRequest(Message):
    __slots__ = ()


class StopRequest(Message):
    __slots__ = ()


class PauseRequest(Message):
    __slots__ = ()


class ExitRequest(Message):
    __slots__ = ()


class StateRequest(Message):
    __slots__ = ()


class StateResponse(Message):
    __slots__ = ('state',)


class GetParametersRequest(Message):
    __slots__ = ()


class GetParametersResponse(Message):
    __slots__ = ('parameter_state',)


class SetParameterRequest(Message):
    __slots__ = ('child_path', 'value')


class SetParameterResponse(Message):
    __slots__ = ('success',)


class UnknownResponse(Message):
    __slots__ = ()


class ExceptionResponse(Message):
    __slots__ = ('exception',)


class MeasureRequest(Message):
    __slots__ = ('position',)


class MeasureResponse(Message):
    __slots__ = ('success',)


class ConnectRequest(Message):
    __slots__ = ()


class ConnectResponse(Message):
    __slots__ = ('state',)


class PushGraphsRequest(Message):
    __slots__ = ('graphs',)


class PullGraphsRequest(Message):
    __slots__ = ()


class GraphsResponse(Message):
    __slots__ = ('graphs',)


class ReplayRequest(Message):
    __slots__ = ('positions', 'measurements')


class ReplayResponse(Message):
    __slots__ = ('enable',)