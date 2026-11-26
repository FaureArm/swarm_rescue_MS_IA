import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle


class PDController:
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.1
    KP_HEADING = 0.8

    def __init__(self, heading_goal=None):
        self.heading_goal = heading_goal

    def control(self, goal, estimate, grasper, *, desired_velocity=None) -> CommandsDict:
        # TODO write description
        self.goal = goal
        self.position_ = estimate.position_
        self.yaw_ = estimate.yaw_
        self.velocity_ = estimate.velocity_
        self.desired_velocity_ = desired_velocity

        # maintain current orientation if dont know where to head to
        if self.heading_goal is None:
            self.heading_goal = self.yaw_

        new_forward, new_lateral = self.compute_thrust()
        new_grasper = grasper
        new_heading = self.compute_heading()

        return {
            "forward": new_forward,
            "lateral": new_lateral,
            "rotation": new_heading,
            "grasper": new_grasper,
        }

    # to avoid division by zero, wrap angles
    @staticmethod
    def _wrap(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))


    def compute_thrust(self):
        """Functions which computes thrusts along x and y axis at given timestep
                
        Returns:
            np.array(): thrusts along x and y axis), scaled for max value
        """
        ## Movement control law: basic PD for now ##
        ###########
        # we calculate ux and uy, thrusts along x and y axis 
        # in map's frame
         
        if self.desired_velocity_ is None:
            # waypoint based pre-existing direct-to-goal PD control
            # NB might wanna delete later on?
            ux, uy = (
                self.KP_POSITION * (self.goal - self.position_)
                - self.KD_VELOCITY * self.velocity_
            )
        else:
            # path following controller
            # track the velocity requested by the path follower
            ux, uy = self.KD_VELOCITY * (
                self.desired_velocity_ - self.velocity_
            )
        ###########

        ## convert our thrusts in drone frame ##
        ###########
        # cosine and sine of our current heading
        cosine, sine = np.cos(self.yaw_), np.sin(self.yaw_)
        # transform into global frame commands
        # TODO: could probably do matrix operation later
        forward = cosine * ux + sine * uy
        lateral = -sine * ux + cosine * uy
        # scale/filter when goes behond max value
        scale = max(1.0, np.hypot(forward, lateral))
        ###########

        return np.array([forward / scale , lateral / scale])


    def compute_heading(self):
        """Functions which computes heading
                        
        Returns:
            np.array(): heading in radians
        """
        ## heading/rotation control law: PD for now ## 
        ###########
        heading_error = normalize_angle(self.heading_goal - self.yaw_)
        heading = np.clip(self.KP_HEADING * heading_error, -1.0, 1.0)
        ###########
        return heading

