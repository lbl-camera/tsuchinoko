from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import Tuple, Iterable, Set, List, Union, Dict, Any, Iterator, Optional

from loguru import logger
from pyqtgraph.parametertree import Parameter

from tsuchinoko.graphs import Graph
from tsuchinoko.utils.mutex import RWLock


@dataclass
class Data:
    """
    A data class to track the data state of an experiment. This type can be appended to as new data is received. A
    locking mechanism is provided to support read/write locking for parallelism.
    """

    dimensionality: int = None
    positions: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    variances: list = field(default_factory=list)
    metrics: dict = field(default_factory=lambda: defaultdict(list))
    states: dict = field(default_factory=dict)
    graphics_items: dict = field(default_factory=dict)

    @property
    def measurements(self) -> List[Tuple[tuple, float, float, dict]]:
        """Get all measurements as list of (position, score, variance, metrics) tuples."""
        return list(zip(self.positions, self.scores, self.variances, [{key: values[i] for key, values in self.metrics.items()} for i in range(len(self))]))

    def __post_init__(self):
        self._lock = RWLock()
        self.w_lock = self._lock.w_locked
        self.r_lock = self._lock.r_locked
        self._completed_iterations = 0

    def inject_new(self, data: List[Tuple[tuple, float, float, Dict[str, Any]]]) -> None:
        """Add new measurements to the data collection.

        Thread-safe method that appends new measurements while holding
        the write lock.

        Args:
            data: List of (position, score, variance, metrics_dict) tuples
        """
        with self.w_lock():
            for datum in data:
                self.positions.append(datum[0])
                self.scores.append(datum[1])
                self.variances.append(datum[2])
                for metric in datum[3]:  # TODO: handle logical cases
                    if metric not in self.metrics:
                        self.metrics[metric] = []
                    self.metrics[metric].append(datum[3][metric])

    def as_dict(self) -> Dict[str, Any]:
        """Convert data to a dictionary for serialization.

        Returns:
            Dictionary with all dataclass fields, metrics converted from
            defaultdict to regular dict.
        """
        self_copy = copy(self)
        self_copy.metrics = dict(self_copy.metrics)
        return asdict(self_copy)

    def __getitem__(self, item: Union[slice, str]) -> Union['Data', List[Any]]:
        """Access data by slice or metric/state name.

        Args:
            item: Either a slice for data subsetting, or a string key
                for accessing metrics, states, or built-in arrays.

        Returns:
            For slices: New Data instance with subset of measurements
            For strings: The requested metric, state, or array

        Raises:
            ValueError: If string key exists in both metrics and states,
                or if key is not found
        """
        if isinstance(item, str):
            if item in self.metrics and item in self.states:
                raise ValueError(f'{item} exists in both states and metrics.')
            elif item in self.metrics:
                return self.metrics[item]
            elif item in self.states:
                return self.states[item]
            elif item.lower() in ['variances', 'scores', 'positions']:
                return getattr(self, item.lower())
        elif isinstance(item, slice):
            if item.stop is None or item.stop > len(self):
                item = slice(item.start, len(self), item.step)
            return Data(self.dimensionality,
                        self.positions[item],
                        self.scores[item],
                        self.variances[item],
                        {key: value[item] for key, value in self.metrics.items()},
                        self.states,
                        self.graphics_items)
        raise ValueError(f'Unknown item: {item}')

    def __contains__(self, item: str) -> bool:
        """Check if a key exists in states or metrics.

        Args:
            item: Key to check for

        Returns:
            True if key exists in states or metrics
        """
        if isinstance(item, str):
            return item in self.states or item in self.metrics
        return False

    def __setitem__(self, key: str, value: List[Any]) -> None:
        """Set a metric array by key.

        Args:
            key: Metric name
            value: List of metric values

        Raises:
            ValueError: If key is not a string
        """
        if isinstance(key, str):
            self.metrics[key] = value
        else:
            raise ValueError()

    def __len__(self) -> int:
        """Return the number of measurements."""
        return len(self.positions)

    def extend(self, data: 'Data') -> None:
        """Extend this data with measurements from another Data instance.

        Thread-safe method that appends all measurements from data while
        holding the write lock. Also updates dimensionality, graphics_items,
        and states from the source data.

        Args:
            data: Data instance to extend from
        """
        with self.w_lock():
            self.positions += data.positions
            self.scores += data.scores
            self.variances += data.variances
            for key in set(self.metrics) | set(data.metrics):
                self.metrics[key] += data.metrics.get(key, [])
            self.dimensionality = data.dimensionality
            self.graphics_items.update(data.graphics_items)
            self.states = data.states

    def __enter__(self) -> 'Data':
        self.w_lock().__enter__()
        logger.exception(RuntimeError("deprecation in progress"))
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Any) -> None:
        self.w_lock().__exit__(exc_type, exc_val, exc_tb)

    def __bool__(self) -> bool:
        """Return True if data contains any measurements."""
        return bool(len(self))

    @contextmanager
    def iteration(self) -> Iterator[None]:
        """Context manager for tracking completed iterations.

        Increments _completed_iterations counter on context exit.
        Used by Core to track experiment progress.

        Yields:
            None
        """
        yield
        self._completed_iterations += 1


class Engine(ABC):
    """
    The Adaptive Engine base class. This component is generally to be responsible for determining future measurement targets.
    """

    dimensionality: int = None
    parameters: Parameter = None
    graphs: List['Graph'] = None
    last_position: tuple = None

    @abstractmethod
    def update_measurements(self, data: Data):
        """
        Update internal variables with the provided new data
        """
        ...

    @abstractmethod
    def request_targets(self, position: Tuple) -> Iterable[Tuple]:
        """
        Determine new targets to be measured

        Parameters
        ----------
        position: tuple
            The current 'position' of the experiment in the target domain.

        Returns
        -------
        targets: array_like
            The new targets to be measured.
        """
        ...

    @abstractmethod
    def reset(self):
        """
        Called when an experiment stops, or is about to start. Returns the engine to a clean state.
        """
        ...

    @abstractmethod
    def train(self):
        """
        Perform training. This can be short-circuited to only train on every N-th iteration, for example.
        """
        ...

    @abstractmethod
    def update_metrics(self, data: Data):
        """
        Calculates various metrics to drive visualizations for the client. The data object is expected to be mutated to
        include these new values.
        """
        ...
