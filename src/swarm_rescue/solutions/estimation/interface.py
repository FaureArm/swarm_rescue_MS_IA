from typing import Protocol

from swarm_rescue.solutions.contracts.observations import StateObservation
from swarm_rescue.solutions.contracts.state import StateEstimate


class StateEstimator(Protocol):
    """One instance per drone; update once per tick, including missing inputs.

    Return None when estimation fails, or an estimate with unavailable fields.
    A new simulation gets a fresh instance through assembly.
    """

    def update(self, observation: StateObservation) -> StateEstimate | None: ...
