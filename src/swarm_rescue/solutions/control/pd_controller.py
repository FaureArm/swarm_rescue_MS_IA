import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.solutions.contracts.motion import MotionCommand, MotionReference
from swarm_rescue.solutions.contracts.state import StateEstimate


class PDController:
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.30
    KP_HEADING = 0.8

    def __init__(self, heading_goal=None):
        self.heading_goal = heading_goal

    def control(self, goal, estimate, grasper) -> CommandsDict:
        command = self.compute_commands(
            MotionReference(goal),
            StateEstimate(estimate.position_, estimate.yaw_, estimate.velocity_),
        )
        return {
            "forward": command.forward,
            "lateral": command.lateral,
            "rotation": command.rotation,
            "grasper": grasper,
        }

    def compute_commands(
        self, reference: MotionReference, estimate: StateEstimate,
    ) -> MotionCommand:
        # TODO write description
        self.goal = reference.destination
        self.position_ = estimate.position
        self.yaw_ = estimate.yaw
        self.velocity_ = estimate.velocity

        # maintain current orientation if dont know where to head to
        if self.heading_goal is None:
            self.heading_goal = self.yaw_

        new_forward = self.compute_thrust()[0]
        new_lateral = self.compute_thrust()[1]
        new_heading = self.compute_heading()

        return MotionCommand(new_forward, new_lateral, new_heading)

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
        #TODO: wwriting full thing for ease of reading, otherwise just following line:
        # ux, uy = self.KP_POSITION * (self.goal - self.position_) - self.KD_VELOCITY * self.velocity_
        vx = self.velocity_[0]
        vy = self.velocity_[1]
        x, y = self.position_[0], self.position_[1]
        goal_x, goal_y = self.goal[0], self.goal[1]
        ux = self.KP_POSITION * (goal_x - x) - self.KD_VELOCITY * vx
        uy = self.KP_POSITION * (goal_y - y) - self.KD_VELOCITY * vy
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

