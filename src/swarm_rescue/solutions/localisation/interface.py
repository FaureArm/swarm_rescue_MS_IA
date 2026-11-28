from typing import Protocol

from ..data_types import Observations, StateEstimate


class EstimationContext(Protocol):
    @property
    def tick_observations(self) -> Observations: ...


class Localisation(Protocol):
    def update(self, drone: EstimationContext) -> StateEstimate | None:
        """Return a complete usable estimate, or None until one is available."""
        ...
