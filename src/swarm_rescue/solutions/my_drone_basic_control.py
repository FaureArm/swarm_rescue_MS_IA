import numpy as np
import arcade
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle


class MyDroneBasicControl(DroneAbstract):
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

    def compute_grasper(self):
        return 0
    
    def control(self):

        self.uptade_goal()
        
        ## scenario where no instructions ##
        ##########
        # if we cant manage to figure out our pos, or dont know where to go
        # then we do nothing  
        # TODO: that means drifting, ideally we would stop
        if self.goal is None or not self.update_estimate():
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # maintain current orientation if dont know where to head to
        if self.heading_goal is None:
            self.heading_goal = self.yaw_
        ###########

        new_forward = self.compute_thrust()[0]
        new_lateral = self.compute_thrust()[1]
        new_grasper = self.compute_grasper()
        new_heading = self.compute_heading()
        

        return {
            "forward": new_forward,
            "lateral": new_lateral,
            "rotation": new_heading,
            "grasper": new_grasper,
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
    