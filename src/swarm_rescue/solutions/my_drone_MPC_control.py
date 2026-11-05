import numpy as np
import arcade
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle


class MyDroneMPCControl(DroneAbstract):
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.30
    KP_HEADING = 0.8
    MPC_HORIZON = 5

    def __init__(self, *, goal=None, heading_goal=None, **kwargs):
        super().__init__(**kwargs)
        # if no goal set, return to center of map
        self.goal = np.asarray((0.0,0.0), dtype=float) if goal is None else np.asarray(goal, dtype=float)
        self.heading_goal = heading_goal
        self.path_ = None
        self.horizon_ = None


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
        return self.measured_gps_position()
        #return np.array(self.true_position(), dtype=float)

    def update_yaw(self):
        """Update yaw

        Returns:
            float: the yaw in radians
        """
        return self.measured_compass_angle()
        #return float(self.true_angle())

    def update_velocity(self):
        """Update velocity
        
        Returns:
            np.array(): x,y velocity
        """
        return self.measured_velocity()
        #return np.array(self.true_velocity(), dtype=float)

    def update_angular_velocity(self):
        """Update angular velocity

        Returns:
            float: the angular velocity in radians per second
        """
        return self.measured_angular_velocity()
        #return float(self.true_angular_velocity())

    def update_estimate(self):
        """Functions which updates or not the state variables of our drone

        Returns:
            boolean: True if we managed to update our state, False else
        """
        self.position_ = self.update_GPS()
        self.yaw_ = self.update_yaw()
        self.velocity_ = self.update_velocity()
        return True

    def update_goal(self):
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

    def global2drone(self, ux_, uy_):
        """Functions which transforms global position to drone position

        Returns:
            np.array(): drone position in global frame
        """
        ## convert our thrusts in drone frame ##
        ###########
        # cosine and sine of our current heading
        cosine, sine = np.cos(self.yaw_), np.sin(self.yaw_)
        # transform into global frame commands
        # TODO: could probably do matrix operation later
        forward = cosine * ux_ + sine * uy_
        lateral = -sine * ux_ + cosine * uy_

        return forward, lateral

    def compute_thrust(self, reference):
        """Functions which computes thrusts along x and y axis at given timestep
                
        Returns:
            np.array(): thrusts along x and y axis), scaled for max value
        """

        # init the state vector for the MPC (current post and velocity)
        current_state = np.concatenate((self.position_, self.velocity_))


        ## Thrust calculation step ##
        ux, uy = self.solve_mpc(current_state, reference)


        ## gloabl frame to drone frame
        forward, lateral = self.global2drone(ux,uy)
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

    def build_path_reference(self):
        # TODO
        return None

    def build_reference(self):
        """what the MPC should aim for over the next N steps

        Returns:
            np.array((4, N+1)): returns the array of reference states for the MPC over the next N steps
            N is horizon size, TODO make it get value from our attribute 
        """
        if self.path is not None:
            # TODO: to sample along the path, when a path actually exists
            return self.build_path_reference()

        # Waypoint mode: every column is [goal_x, goal_y, 0, 0] (we want speed = 0 when we get there thus the zeroes)
        goal_state = np.array([self.goal[0], self.goal[1], 0.0, 0.0])
        return np.repeat(goal_state[:, None], self.horizon + 1, axis=1)

    def predict_step(self):
        return
    
    def control(self):

        self.update_goal()
        
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

        
        reference = self.build_reference()
        new_forward, new_lateral = self.compute_thrust(reference)
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
    