from typing import Protocol

from ..data_types import Observations, StateEstimate

"""Define what this subsystem can read and what it must provide.

These protocols document the required methods, inputs, and return values.
Any implementation that follows these contracts can be swapped in through
assembly.py without changing the code that calls it.
Implementations must provide the methods defined here and return the specified values.
"""

class EstimationContext(Protocol):
    @property
    def tick_observations(self) -> Observations: ...


class Localisation(Protocol):
    def update(self, drone: EstimationContext) -> StateEstimate | None:
        """Return a complete usable estimate, or None until one is available."""
        ...
