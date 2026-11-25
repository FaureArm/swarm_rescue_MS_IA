import numpy as np
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor


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

class SemanticState:
    def __init__(self):
        self.has_found_wounded = False
        self.has_grasped_wounded = False 
        self.has_found_rescue_center = False
        self.wounded_list = []

    def process_semantic_data(self, estimator:MeasuredState, detection_semantic):
        """
        According to his state in the state machine, the Drone will move
        toward a wounded person or the rescue center
        """
        has_found_wounded = False
        has_grasped_wounded = False 
        has_found_rescue_center = False
        wounded_list = []

        if detection_semantic is not None:
            for data in detection_semantic:
                # If the wounded person detected
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    # If the wounded person is grasped
                    if data.grasped:
                        print("grasped")
                        has_grasped_wounded = True
                    print("not grasped")
                    print(data)
                        
                    has_found_wounded = True

                    # PM TODO : compute a score to affect victim to a drone in case of multiple drone and multiple victim
                    # PM compute victim position from drone position
                    wounded_position = np.array([estimator.position_[0] + data.distance * np.cos(estimator.yaw_ + data.angle), 
                                                 estimator.position_[1] + data.distance * np.sin(estimator.yaw_ + data.angle)])

                    wounded_list.append(wounded_position)


                if (has_found_wounded and has_grasped_wounded and 
                        data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                    has_found_rescue_center = True


        self.has_found_wounded = has_found_wounded if self.has_found_wounded is not has_found_wounded else self.has_found_wounded
        self.has_grasped_wounded = has_grasped_wounded if self.has_grasped_wounded is not has_grasped_wounded else self.has_grasped_wounded
        self.has_found_rescue_center = has_found_rescue_center if self.has_found_rescue_center is not has_found_rescue_center else self.has_found_rescue_center
        self.wounded_list = wounded_list if self.wounded_list is not wounded_list else self.wounded_list

        return True

    def update_semantic(self, has_found_wounded, has_found_rescue_center, wounded_list):
        """Store the semantic detection"""
        self.has_found_wounded = has_found_wounded
        self.has_found_rescue_center = has_found_rescue_center
        self.wounded_list = wounded_list