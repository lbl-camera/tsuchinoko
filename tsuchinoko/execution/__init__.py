from abc import ABC, abstractmethod
from typing import Tuple, List


class Engine(ABC):
    """
    The Execution Engine base class. This component is generally to be responsible for measuring targets.
    """

    @abstractmethod
    def update_targets(self,  targets: List[Tuple]):
        """
        Updates the list of measurement targets. Generally new targets will take precedence, and will be stashed for
        measurement in the background.

        Parameters
        ----------
        targets: list
            A list of new measure target positions.
        """
        ...

    @abstractmethod
    def get_position(self) -> Tuple:
        """
        Returns the current position in the target domain.

        Returns
        -------
        current_position: tuple
            The current position in the target domain

        """
        ...

    @abstractmethod
    def get_measurements(self) -> List[Tuple]:
        """
        Returns new measurements made since its last call. Generally measured values are accumulated by a background thread,
        then returned and cleared here.

        Returns
        -------
        position: tuple
            The measured target position.
        value: float
            The value at that position.
        variance: float
            The variance associated with the measurement of that value.
        metrics: dict
            Any non-standard metric values to be used for visualization at the client.
        """
        ...
