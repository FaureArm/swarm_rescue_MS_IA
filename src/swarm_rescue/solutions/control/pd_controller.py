import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.utils import normalize_angle
from swarm_rescue.solutions.data_types import MissionDecision, Activity, MotionReference, StateEstimate
from .interface import ControlContext


class PDController:
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.30
    KP_HEADING = 0.8

    def __init__(self, heading_goal=None):
        self.heading_goal = heading_goal

    def update(self, drone: ControlContext) -> CommandsDict:
        return self._commands(
            drone.navigation_result.motion, drone.pose, drone.intent.grasper,
            face_target=drone.intent.face_target,
        )

    def control(self, decision: MissionDecision, estimate) -> CommandsDict:
        """Compatibility entry point for the standalone planning scenario."""
        pose = StateEstimate(estimate.position_, estimate.yaw_, estimate.velocity_)
        standby = decision.role is Activity.STAND_BY
        return self._commands(
            None if standby else decision.motion_reference, pose,
            0 if standby else decision.grasper,
            face_target=decision.role in (Activity.GRASPING_WOUNDED, Activity.DROPPING_AT_RESCUE_CENTER),
        )

    def _commands(
        self, motion: MotionReference | None, pose: StateEstimate | None,
        grasper: int, *, face_target: bool,
    ) -> CommandsDict:
        if pose is None or motion is None or motion.target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": grasper}

        self.goal = motion.target
        self.position_ = pose.position
        self.yaw_ = pose.yaw
        self.velocity_ = pose.velocity
        self.desired_velocity_ = motion.velocity

        # maintain current orientation if dont know where to head to
        if self.heading_goal is None:
            self.heading_goal = self.yaw_

        forward, lateral = self.compute_thrust()
        if face_target:
            rotation = self.compute_frontal_approach()
        elif motion.heading is not None:
            rotation = np.clip(self.KP_HEADING * normalize_angle(motion.heading - pose.yaw), -1.0, 1.0)
        else:
            rotation = self.compute_heading()
        return {
            "forward": float(forward), "lateral": float(lateral),
            "rotation": float(rotation), "grasper": grasper,
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



    def compute_frontal_approach(self):
        """Functions which computes rotation to face the goal

        Returns:
            np.array(): heading in radians
        """
        x, y = self.position_[0], self.position_[1]
        goal_x, goal_y = self.goal[0], self.goal[1]
        ux = (goal_x - x)
        uy = (goal_y - y)

        direction = np.atan2(uy, ux)
        diff_angle = normalize_angle(direction - self.yaw_)

        stop_angle = 0.1
        a = abs(diff_angle) - stop_angle

        if a < self.KP_POSITION:
            # stop rotation when the yaw target is reached
            rotation = 0
        else:
            # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
            # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
            rotation = 2 / (1 + np.exp(-diff_angle / (stop_angle * 0.5))) - 1

        return rotation

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
