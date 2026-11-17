import math
from enum import Enum

import arcade
import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.utils import circular_mean, normalize_angle


class MyDroneControlAndRescue(DroneAbstract):
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.30
    KP_HEADING = 0.8

    # wounded position in map 1
    wounded_person_position = [-310, -180]
    # path points to victim in map 1
    exploration_path = list
    must_initialize_rescue_center: bool
    rescue_center_position = [0, 0]
    current_goal_index: int
    exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]

    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """

        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    class Wall_Position(Enum):
        """
        Different wall position regarding the drone attitude
        Can be a combination
        """

        NO_WALL = 0
        AHEAD = 1
        BEHIND = 2
        LEFT = 3
        RIGHT = 4

    def __init__(self, *, goal=None, heading_goal=None, **kwargs):
        super().__init__(**kwargs)
        
        # if no goal set, return to center of map
        self.goal = (np.asarray(self.exploration_path[0], dtype=float) 
                     if goal is None 
                     else np.asarray(goal, dtype=float))
        self.heading_goal = heading_goal

        self.position_ = None
        self.yaw_ = None
        self.velocity_ = np.zeros(2)

        self.wall_position = self.Wall_Position.NO_WALL
        self.state = self.Activity.SEARCHING_WOUNDED
        self.must_initialize_rescue_center = True
        self.current_goal_index = 0

    # to avoid division by zero, wrap angles
    @staticmethod
    def _wrap(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def define_message_for_all(self) -> None:
        """
        Laucnher requires this method to be declared
        """
        pass

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        toward a wounded person or the rescue center
        """
        command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (
            self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED
        ) and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (
                    data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON
                    and not data.grasped
                ):
                    found_wounded = True
                    v = (data.angle * data.angle) + (data.distance * data.distance / 10**5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (
            self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER
        ) and detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = data.distance < 30

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0.0
            command["rotation"] = -1.0

        return found_wounded, found_rescue_center, command

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

    def uptade_goal(self, reverse:bool = False):
        """Functions which updates or not the current goal
        
        Returns:
            boolean: True if we managed to update our goal, False else
        """
        # change goals
        if self.goal is not None and self.position_ is not None :
            
            # if the goal is reached within the exploration path, step to the next one
            if np.linalg.norm(self.goal - self.position_) < 10:
                if reverse:
                    self.current_goal_index -=1
                else:
                    self.current_goal_index +=1

                if (self.current_goal_index < len(self.exploration_path) 
                    and
                    self.current_goal_index >= 0):
                    self.goal = np.asarray(self.exploration_path[self.current_goal_index], dtype=float) 
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

    # def compute_grasper(self):
    #     return 0
    
    def control(self):

        if self.must_initialize_rescue_center:
            self.rescue_center_position = self.measured_gps_position()
            self.exploration_path.insert(0, self.rescue_center_position)
            self.must_initialize_rescue_center = False

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.grasper.grasped_wounded_persons:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif (
            self.state is self.Activity.DROPPING_AT_RESCUE_CENTER
            and not self.grasper.grasped_wounded_persons
        ):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is
        # detected, we use a special command
        ##########
        if self.state is self.Activity.SEARCHING_WOUNDED:
            self.uptade_goal()
            command = self.control_searching_mode()

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            self.uptade_goal(reverse=True)
            command = self.control_searching_mode()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command

    def control_searching_mode(self) -> dict:
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit.
        Returns:
            CommandsDict: The control command for the drone.
        """

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

        return  {
            "forward": self.compute_thrust()[0],
            "lateral": self.compute_thrust()[1],
            "rotation": self.compute_heading(),
            # always set to 0 while exploring
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
    