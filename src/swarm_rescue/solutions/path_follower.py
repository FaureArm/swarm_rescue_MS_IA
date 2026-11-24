"Define path following control law"

from collections.abc import Sequence

import numpy as np

from swarm_rescue.solutions.data_types import MotionReference, Waypoint
from swarm_rescue.solutions.localisation import MeasuredState

class PathFollower:
    def __init__(self, path, lookahead, gain, max_pos, max_vel):
        self.path_ = list(path) # path of waypoints to follow, final waypoint is final destination
        self.lookahead_ = lookahead # lookahead in pixels, could also implement in nber of waypoints/index?
        self.position_gain_  = gain # 
        self.max_pos_ = max_pos # maximum positon tolerance for arrival # TODO change name?
        self.max_vel_ = max_vel # maximum velocity at arrival #TODO same change name?
        self.index = 0

        self.positions = np.array(
            [(waypoint.x, waypoint.y) for waypoint in self.path_], dtype=float
        )
        self.speeds = np.array(
            [waypoint.velocity for waypoint in self.path_], dtype=float
        )
        self.headings = [
            None if waypoint.heading is None else float(waypoint.heading)
            for waypoint in self.path_
        ]

    def update(self, estimate) -> MotionReference:
        """path following module update function. At each uptade,
        checks the current waypoint to follow in path, and returns
        control targets

        Args:
            estimate (_type_): Drone current pose

        Returns:
            MotionReference: return to control module, what should currently target
            at timestep t
        """
        # need current drone position to check against
        # where we are in the path
        position = np.asarray(estimate.position_, dtype=float)

        while True:
            # we calculate the segment between our two next waypoints
            start = self.positions[self.index]
            end = self.positions[self.index + 1]
            segment = end - start

            # check if we have passed the current waypoint
            # by verifying our progression by projecting our pos
            # against the segment between our two next waypoints
            fraction = np.dot(position - start, segment) / np.dot(segment, segment)
            if (
                fraction < 1.0
                or self.index == len(self.positions) - 2
            ):
                break
            # if we pass the segment, then we are at the next waypoint
            # so we update our index
            self.index += 1

        ### Now we need to figure out the desired velocity and pos###
        
        #  find the closest point on our segment
        projection = start + np.clip(fraction, 0.0, 1.0) * segment
        target = self._lookahead_point(projection)

        # calculate the euclidean distance bewteen
        # the drone's current pos and the current target
        displacement = target - position
        distance = np.linalg.norm(displacement)

        # get the target speed
        # cap speed at the assigned value
        # and slow down near the target
        speed = min(
            self.path_[self.index].velocity, #TODO: for now just use previous waypoint speed, either pick next but issue with end point, or gradual change?
            self.position_gain_ * distance,
        )

        # derive an actual velocity vector
        velocity = np.zeros(2)
        if distance > 0:
            velocity = speed * displacement / distance

        # return for controller
        return MotionReference(
            position=target,
            velocity=velocity,
            heading=None,
        )

    def _lookahead_point(self, projection):
        remaining = self.lookahead_
        start = projection

        # Walk forward along segments until the lookahead distance is used.
        for index in range(self.index, len(self.positions) - 1):
            end = self.positions[index + 1]
            segment = end - start
            length = np.linalg.norm(segment)

            if length > remaining:
                return start + remaining * segment / length

            remaining -= length
            start = end

        return self.positions[-1].copy()