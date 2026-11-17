"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""

import math
from enum import Enum
from typing import Optional, Tuple, Dict

import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import circular_mean, normalize_angle


class MySmartDrone(DroneAbstract):
    """
    Drone controller

    Attributes:
        counterStraight (int): Counter for straight movement.
        angleStopTurning (float): Angle to stop turning.
        distStopStraight (float): Distance to stop moving straight.
        isTurning (bool): Whether the drone is currently turning.
    """

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

    stop_rotation: bool
    stop_forward: bool
    wall_position: Wall_Position

    EPSILON: float

    # wounded position in map 1
    wounded_person_position = [-310, -180]
    must_initialize_rescue_center: bool
    rescue_center_position = [0, 0]
    current_goal_index: int
    exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]

    def __init__(
        self, identifier: Optional[int] = None, misc_data: Optional[MiscData] = None, **kwargs
    ):
        """
        Initializes the drone controller.

        Args:
            identifier (Optional[int]): Drone identifier.
            misc_data (Optional[MiscData]): Miscellaneous data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            identifier=identifier, misc_data=misc_data, display_lidar_graph=False, **kwargs
        )
        self.EPSILON = 1e-2
        self.stop_rotation = False
        self.stop_forward = False
        self.wall_position = self.Wall_Position.NO_WALL
        self.state = self.Activity.SEARCHING_WOUNDED
        self.must_initialize_rescue_center = True

        self.current_goal_index = 0
        self.goal = np.asarray(self.exploration_path[0], dtype=float)

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def uptade_goal(self, reverse:bool = False):
        """Functions which updates or not the current goal
        
        Returns:
            boolean: True if we managed to update our goal, False else
        """
        # change goals
        if self.goal is not None :
            
            # if the goal is reached within the exploration path, step to the next one
            if np.linalg.norm(self.goal - self.measured_gps_position()) < 20:
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

    def control(self) -> CommandsDict:
        command: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

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

        # print(
        #     f"state={self.state.name:<24}  "
        #     f"can_grasp={self.grasper.can_grasp}  "
        #     f"grasped={self.grasper.grasped_wounded_persons}"
        # )

        ##########
        # COMMANDS FOR EACH STATE
        # Searching randomly, but when a rescue center or wounded person is
        # detected, we use a special command
        ##########
        if self.state is self.Activity.SEARCHING_WOUNDED:
            self.uptade_goal()
            command = self.control_to_goal()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            self.uptade_goal(reverse=True)
            command = self.control_to_goal()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command

    def process_lidar_sensor(self, the_lidar_sensor) -> Tuple[Dict[str, float], bool]:
        """
        Processes the lidar sensor to avoid obstacles.

        Args:
            the_lidar_sensor: The lidar sensor object.

        Returns:
            Tuple[Dict[str, float], bool]: The command and collision status.
        """
        command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return command, False

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution

        if size != 0:
            min_dist = min(values)
            # near_angle_raw: angle with the nearest distance
            index_ray = np.argmin(values)
            near_angle = ray_angles[index_ray]

            if 0 < index_ray < 10 or 180 < index_ray < 190:
                self.is_wall_behind = True
            elif 70 < index_ray < 110:
                self.is_wall_behind = False
                self.is_wall_on_left = False

            # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
            intensity = 2 / (1 + math.exp(-min_dist / 30)) - 1
            command["forward"] = min(intensity, 0.7)

            heading = normalize_angle(near_angle - self.measured_compass_angle())
            dx = intensity * math.cos(heading)
            dy = intensity * math.sin(heading)

            print(f"dx: {dx}, dy :{dy}")

            command["forward"] = dx
            if self.is_wall_on_left:
                command["lateral"] = dy
            else:
                command["lateral"] = -dy

        return command

    def control_explore(self) -> CommandsDict:
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit.
        Returns:
            CommandsDict: The control command for the drone.
        """

        command = self.process_lidar_sensor(self.lidar())

        return command

    def control_cheat(self) -> CommandsDict:
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit.
        Returns:
            CommandsDict: The control command for the drone.
        """

        command: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        if not self.stop_forward or not self.stop_rotation:
            dx = self.wounded_person_position[0] - self.measured_gps_position()[0]
            dy = self.wounded_person_position[1] - self.measured_gps_position()[1]
            distance = math.sqrt(dx**2 + dy**2)

            direction = math.atan2(dy, dx)
            diff_angle = normalize_angle(direction - self.measured_compass_angle())

            stop_angle = 0.1
            a = abs(diff_angle) - stop_angle

            if a < self.EPSILON:
                # stop drone when the position is reached
                command["rotation"] = 0
                self.stop_rotation = True

            else:
                self.stop_rotation = False
                # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
                # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
                rotation_intensity = 2 / (1 + math.exp(-diff_angle / (stop_angle * 0.5))) - 1
                command["rotation"] = rotation_intensity

            desired_dist = 5
            d = distance - desired_dist

            if d < self.EPSILON:
                # stop drone when the position is reached
                command["forward"] = 0
                self.stop_forward = True

            else:
                self.stop_forward = False
                # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
                # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
                if self.state is not self.Activity.SEARCHING_RESCUE_CENTER:
                    intensity = 2 / (1 + math.exp(-d / d*0.6)) - 1
                else:
                    # full speed for recovering
                    intensity = 1
                    
                command["forward"] = min(intensity, 0.7)

        return command

    def control_to_rescue(self) -> CommandsDict:
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit.
        Returns:
            CommandsDict: The control command for the drone.
        """

        command: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        if not self.stop_forward or not self.stop_rotation:
            dx = self.rescue_center_position[0] - self.measured_gps_position()[0]
            dy = self.rescue_center_position[1] - self.measured_gps_position()[1]
            distance = math.sqrt(dx**2 + dy**2)

            direction = math.atan2(dy, dx)
            diff_angle = normalize_angle(direction - self.measured_compass_angle())

            stop_angle = 0.1
            a = abs(diff_angle) - stop_angle

            if a < self.EPSILON:
                # stop drone when the position is reached
                command["rotation"] = 0
                self.stop_rotation = True

            else:
                self.stop_rotation = False
                # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
                # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
                rotation_intensity = 2 / (1 + math.exp(-diff_angle / (stop_angle * 0.5))) - 1
                command["rotation"] = rotation_intensity

            desired_dist = 5
            d = distance - desired_dist

            if d < self.EPSILON:
                # stop drone when the position is reached
                command["forward"] = 0
                self.stop_forward = True

            else:
                self.stop_forward = False
                # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
                # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
                intensity = 2 / (1 + math.exp(-d / 300)) - 1
                command["forward"] = min(intensity, 0.7)

        return command

    def control_to_goal(self) -> CommandsDict:
        """
        The Drone will move forward and turn for a random angle when an obstacle is hit.
        Returns:
            CommandsDict: The control command for the drone.
        """

        command: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        if not self.stop_forward or not self.stop_rotation:
            dx = self.goal[0] - self.measured_gps_position()[0]
            dy = self.goal[1] - self.measured_gps_position()[1]
            distance = math.sqrt(dx**2 + dy**2)

            direction = math.atan2(dy, dx)
            diff_angle = normalize_angle(direction - self.measured_compass_angle())

            stop_angle = 0.1
            a = abs(diff_angle) - stop_angle

            if a < self.EPSILON:
                # stop drone when the position is reached
                command["rotation"] = 0
                self.stop_rotation = True

            else:
                self.stop_rotation = False
                # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
                # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
                rotation_intensity = 2 / (1 + math.exp(-diff_angle / (stop_angle * 0.5))) - 1
                command["rotation"] = rotation_intensity

            desired_dist = 5
            d = distance - desired_dist

            if d < self.EPSILON:
                # stop drone when the position is reached
                command["forward"] = 0
                self.stop_forward = True

            else:
                self.stop_forward = False
                # We use a logistic function. -1 < intensity(d) < 1 and  intensity(0) = 0
                # intensity(d) approaches 1 (resp. -1) as d approaches +inf (resp. -inf)
                if self.state is not self.Activity.SEARCHING_RESCUE_CENTER:
                    intensity = 2 / (1 + math.exp(-d / d*0.6)) - 1
                else:
                    # full speed for recovering
                    intensity = 1
                command["forward"] = min(intensity, 0.7)

        return command
