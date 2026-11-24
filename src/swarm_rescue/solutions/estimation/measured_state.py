import numpy as np

from swarm_rescue.solutions.contracts.observations import StateObservation
from swarm_rescue.solutions.contracts.state import StateEstimate


class MeasuredState:
    def __init__(self):
        self.position_ = None
        self.yaw_ = None
        self.velocity_ = np.zeros(2)

    def update_estimate(self, position, yaw, velocity):
        """Store the GPS, compass and velocity readings for this tick."""
        self.position_ = position
        self.yaw_ = yaw
        self.velocity_ = velocity
        return True

    def update(self, observation: StateObservation) -> StateEstimate | None:
        if not self.update_estimate(
            observation.position, observation.yaw, observation.velocity
        ):
            return None
        return StateEstimate(self.position_, self.yaw_, self.velocity_)
