import numpy as np
from swarm_rescue.solutions.kalman import KalmanFusion
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor



class MeasuredState:
    def __init__(self):
        self._kf = KalmanFusion()
        self.position_ = None
        self.yaw_ = None
        self.velocity_ = np.zeros(2)

    def update_estimate(self, position, yaw, velocity):
        """Store the GPS, compass and velocity readings for this tick."""
        self.yaw_ = yaw

        # Application filtre Kalman
        est = self._kf.step(position, velocity)
        self.position_ = np.asarray(est[:2])
        self.velocity_ = np.asarray(self._kf.velocity())
        return True


class SemanticState:
    def __init__(self):
        self.has_found_wounded = False
        self.wounded_position_list = []
        self.rescue_center_position = None

    def compute_POI_position(self, estimator:MeasuredState, semantic_data) -> tuple:
        """_summary_

        Args:
            estimator (MeasuredState): _description_
            semantic_data (_type_): _description_

        Returns:
            tuple: _description_
        """
        # compute victim position from drone position
        POI_position = np.array([estimator.position_[0] + semantic_data.distance * np.cos(estimator.yaw_ + semantic_data.angle), 
                                estimator.position_[1] + semantic_data.distance * np.sin(estimator.yaw_ + semantic_data.angle)])
        
        return POI_position

    def update_semantic(self, estimator:MeasuredState, semantic_data_table, has_grasped_wounded:bool):
        """
        According to his state, the Drone will move
        toward a wounded person or the rescue center
        """
        self.wounded_position_list = []

        if semantic_data_table is not None:
            for semantic_data in semantic_data_table:

                # If a wounded person is detected
                if (semantic_data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON):

                    # if that's the first time this victim is discovered
                    if not self.has_found_wounded:
                        self.has_found_wounded = True

                    wounded_position = self.compute_POI_position(estimator, semantic_data)

                    # PM TODO: IN CASE OF MANY DETECTED VICTIMS
                    # compute a score to affect a spcific victim to a drone 
                    # in case of multiple drone and multiple victim
                    self.wounded_position_list.append(wounded_position)

                elif (semantic_data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and
                      self.rescue_center_position is None):
                    self.rescue_center_position = self.compute_POI_position(estimator, semantic_data)

        # PM: no more wounded identified - back to exploration
        if len(self.wounded_position_list) == 0:
            self.has_found_wounded = False

        return True