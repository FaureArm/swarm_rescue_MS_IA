import numpy as np
import arcade
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle


class MyDroneBadControl(DroneAbstract):
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.30
    KP_HEADING = 0.8

    def __init__(self, *, goal=None, heading_goal=None, **kwargs):
        super().__init__(**kwargs)
        # if no goal set, return to center of map
        self.goal = np.asarray((0.0,0.0), dtype=float) if goal is None else np.asarray(goal, dtype=float)
        self.heading_goal = heading_goal

        self.position_ = None
        self.yaw_ = None
        self.velocity_ = np.zeros(2)

    # to avoid division by zero, wrap angles
    @staticmethod
    def _wrap(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def define_message_for_all(self) -> None:
        """
        Laucnher requires this method to be declared
        """
        pass

    def update_GPS(self):
        """Update GPS
        
        Returns:
            np.array(): x,y GPS position
        """
        # return self.measured_gps_position()
        return np.array(self.true_position(), dtype=float)

    def update_yaw(self):
        """Update yaw

        Returns:
            float: the yaw in radians
        """
        # return self.measured_compass_angle()
        return float(self.true_angle())

    def update_velocity(self):
        """Update velocity
        
        Returns:
            np.array(): x,y velocity
        """
        # return self.measured_velocity()
        return np.array(self.true_velocity(), dtype=float)

    def update_angular_velocity(self):
        """Update angular velocity

        Returns:
            float: the angular velocity in radians per second
        """
        # return self.measured_angular_velocity()
        return float(self.true_angular_velocity())

    def update_estimate(self):
        """Functions which updates or not the state variables of our drone

        Returns:
            boolean: True if we managed to update our state, False else
        """
        self.position_ = self.update_GPS()
        self.yaw_ = self.update_yaw()
        self.velocity_ = self.update_velocity()
        return True

    def uptade_goal(self):
        """Functions which updates or not the current goal
        
        Returns:
            boolean: True if we managed to update our goal, False else
        """
        # change goals
        if self.goal is not None and self.position_ is not None :
            if np.linalg.norm(self.goal - self.position_) < 0.5:
                self.goal = np.random.uniform(-100, 100, size=2)  # random new goal if close enough
                print(f"New goal: {self.goal}")

        return True

    def compute_thrust(self, distance, heading_error):
        if distance <= 5.0 or abs(heading_error) > np.deg2rad(10.0):
            return 0.0
        return 0.5

    def compute_heading(self, heading_error):
        return float(np.clip(self.KP_HEADING * heading_error, -1.0, 1.0))

    def compute_grasper(self):
        return 0
    
    def control(self):
        if (
        self.goal is None
        or not self.update_estimate()
        or self.position_ is None
        or self.yaw_ is None
        ):
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        self.uptade_goal()
        dx, dy = self.goal - self.position_
        distance = np.hypot(dx, dy)
        rotation = 0.0
        forward = 0.0

        if distance > 5.0:
            desired_heading = np.arctan2(dy, dx)
            heading_error = normalize_angle(desired_heading - self.yaw_)
            rotation = self.compute_heading(heading_error)
            forward = self.compute_thrust(distance, heading_error)

        return {
            "forward": forward,
            "lateral": 0.0,
            "rotation": rotation,
            "grasper": 0,
        }

    #########
    # utils stuff for debug, to remove later
    # could be interesting later on to create a class
    # to draw stuff for our drones
    def draw_top_layer(self):
        if self.goal is None:
            return

        # draw a basic circle
        x, y = self.goal
        arcade.draw_circle_outline(
            center_x=float(x),
            center_y=float(y),
            radius=8,
            color=(255, 0, 255),
            border_width=2,
        )
    