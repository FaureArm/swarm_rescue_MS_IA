import numpy as np
import arcade
import matplotlib.pyplot as plt
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.utils import normalize_angle


class MyDroneMapper(DroneAbstract):
    # gains, to adjust/test -> todo automatically, running headless?
    KP_POSITION = 0.015
    KD_VELOCITY = 0.05
    KP_HEADING = 0.8

    def __init__(self, *, goal=None, heading_goal=None, **kwargs):
        super().__init__(**kwargs)
        # if no goal set, return to center of map
        self.mission_goal = np.asarray((0.0,0.0), dtype=float) if goal is None else np.asarray(goal, dtype=float)
        self.active_waypoint = self.mission_goal
        self.heading_goal = heading_goal
        self.map = set() # map holding hard pixels. Set to easily handle duplicates
        self.avoidance_sign = None

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

    def uptade_mission_goal(self):
        """Functions which updates or not the current goal
        
        Returns:
            boolean: True if we managed to update our goal, False else
        """
        #TODO: add something that checks that goal is not outside map boundaries

        # if final mission goal achieved, then do random
        if self.mission_goal is not None and self.position_ is not None :
            if np.linalg.norm(self.mission_goal - self.position_) < 1:
                self.mission_goal = np.random.uniform(-200, 200, size=2)  # random new goal if close enough
                print(f"New goal: {self.mission_goal}")

        # later could add a condition on either class, or if has a person, or if no coms, etc
        
        self.update_active_waypoint()

        return True

    def update_active_waypoint(self):
        """method to update active waypoint, based on mission
         goal and obstacle presence
        """
        # if we reach active waypoint, we default back to mission goal
        if self.active_waypoint is not None and self.position_ is not None :
            if np.linalg.norm(self.active_waypoint - self.position_) < 30:
                self.active_waypoint = self.mission_goal

        # now we check that there isnt any obs
        # check with a segment if we cross hard pixels, does it come across a hard pixel
        if self.obstacle_toward_waypoint():
            # obstacle present, so we need to find another waypoint

            if self.avoidance_sign is None:
                self.avoidance_sign = float(np.sign(self.mission_goal[1] - self.position_[1])) or 1.0
            start = np.asarray(self.position_, dtype=float)
            candidate = start + np.array([0.0, 50.0 * self.avoidance_sign])

            # World bounds are centered on (0, 0). Leave room for the drone's radius.
            limits = np.asarray(self.size_area, dtype=float) / 2.0 - 12.0

            if np.any(np.abs(candidate) > limits):
                opposite = start + np.array([0.0, -50.0 * self.avoidance_sign])

                if np.all(np.abs(opposite) <= limits):
                    self.avoidance_sign *= -1.0
                    candidate = opposite
                else:
                    candidate = start.copy()  # Neither candidate fits: brake.

            self.active_waypoint = candidate


            """direction = np.asarray(self.mission_goal) - np.asarray(self.position_)
            direction = direction / np.linalg.norm(direction)
            angle_sign = np.sign(np.arctan2(self.mission_goal[1] - self.position_[1], self.mission_goal[0] - self.position_[0]))
            self.active_waypoint = self.position_ + 50.0 * angle_sign"""
            #print(f"Angle sign: {angle_sign}")
            print(f"Avoidance sig: {self.avoidance_sign}")
            print(f"New active waypoint: {self.active_waypoint}")
        else:
            self.avoidance_sign = None
            self.active_waypoint = self.mission_goal
        return True

    def obstacle_toward_waypoint(self):
        """now we check that there isnt any obs
        # check with a segment if we cross hard pixels, does it come across a hard pixel

        Returns:
            _type_: _description_
        """

        if self.position_ is None or self.active_waypoint is None:
            return False

        start = np.asarray(self.position_, dtype=float)
        delta = np.asarray(self.mission_goal, dtype=float) - start
        distance = np.linalg.norm(delta)
        if distance == 0 or not self.map:
            return False

        # Drone radius is 10, so its diameter is 20 world pixels
        length = min(100.0, distance)
        end = start + delta / distance * length

        # Sample densely, then round just like update_lidar_map()
        count = int(np.ceil(length * 2)) + 1
        pixels = np.rint(np.linspace(start, end, count)).astype(int)

        return any((int(x), int(y)) in self.map for x, y in pixels)

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
        # ux, uy = self.KP_POSITION * (self.active_waypoint - self.position_) - self.KD_VELOCITY * self.velocity_
        vx = self.velocity_[0]
        vy = self.velocity_[1]
        x, y = self.position_[0], self.position_[1]
        goal_x, goal_y = self.active_waypoint[0], self.active_waypoint[1]
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

    def update_lidar_map(self, drone_position, yaw):
        """updates the drone's internal map by adding pixels and their immediate
        surrounding pixels as hard pixels in the drone's map. Hard limit of 275 px rays
        NB: could make a better version where rays calculate map-frame coords, drones share these
        which allows us to kalman the shit out of each hard pixel hit to have high confidence in hard px

        Args:
            drone_position (_type_): drone position in worl frame
            yaw (_type_): drone heading in world frame, in radian
        """
        # likely no situation where no lidar values are returned 
        # but just in case -> unless some filtering?
        ranges = self.lidar_values() #TODO: replace with a getter, so that we can apply kalman filtering later
        if ranges is None:
            return

        ranges = np.asarray(ranges)
        angles = np.asarray(self.lidar_rays_angles())

        # limit max ray distance to 275 px, because noise beyond that can make too many errors
        # to change later on if we get better readings
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= 275)

        bearings = yaw + angles[valid]
        # give direction of px hit in a vector: [0, 1] for 90 deg hit for ex
        directions = np.column_stack((np.cos(bearings), np.sin(bearings)))
        # calculate the world frame coords of that hit (direction * length of ray + add drone's world frame pos)
        hits = drone_position + ranges[valid, None] * directions
        pixels = np.rint(hits).astype(int)

        for x, y in pixels:
            # add immediate neighbours to our positive pixel hit
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.map.add((x + dx, y + dy)) #adds coords of pixel. Set auto handles dupes
    
    def control(self):

        self.update_lidar_map(self.true_position(), self.true_angle())

        self.uptade_mission_goal()
        
        ## scenario where no instructions ##
        ##########
        # if we cant manage to figure out our pos, or dont know where to go
        # then we do nothing  
        # TODO: that means drifting, ideally we would stop
        if self.active_waypoint is None or not self.update_estimate():
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
        if self.map:
         arcade.draw_points(list(self.map), (0, 220, 255, 160), 2)
        if self.active_waypoint is None:
            return

        # draw a basic circle
        x_active_waypoint, y_active_waypoint = self.active_waypoint
        arcade.draw_circle_outline(
            center_x=float(x_active_waypoint),
            center_y=float(y_active_waypoint),
            radius=8,
            color=(255, 0, 255),
            border_width=2,
        )

        x_mission_goal, y_mission_goal = self.mission_goal
        arcade.draw_circle_outline(
                center_x=float(x_mission_goal),
                center_y=float(y_mission_goal),
                radius=8,
                color=(0, 0, 255),
                border_width=5,
            )

        if self.position_ is not None and self.active_waypoint is not None:
            start = np.asarray(self.position_, dtype=float)
            delta = np.asarray(self.active_waypoint, dtype=float) - start
            distance = np.linalg.norm(delta)

            if distance > 0:
                end = start + delta / distance * min(20.0, distance)
                arcade.draw_line(
                    float(start[0]), float(start[1]),
                    float(end[0]), float(end[1]),
                    (255, 0, 0), 2,
                )

