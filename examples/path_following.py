"""Manual path-controller experiment on the launcher's first evaluation map."""

import argparse

import arcade
import numpy as np

from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.data_types import Waypoint
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.localisation import MeasuredState
from swarm_rescue.solutions.mission_planner import WaypointMission
from swarm_rescue.solutions.path_follower import PathFollower


# Below the wall ending at y=-93, then above the wall ending at y=89.
# Beveled turns avoid the current follower stalling at exact right angles.
# The final point stays clear of the wounded person's moving path.
PATH = (
    Waypoint(295.0, 118.0, 1.5, None),
    Waypoint(170.0, -165.0, 1.5, None),
    Waypoint(-60.0, -165.0, 1.5, None),
    Waypoint(-100.0, -125.0, 1.5, None),
    Waypoint(-100.0, 125.0, 1.5, None),
    Waypoint(-140.0, 165.0, 1.5, None),
    Waypoint(-280.0, 165.0, 1.5, None),
    Waypoint(-320.0, 125.0, 1.5, None),
    Waypoint(-320.0, -110.0, 0.0, None),
)


def path_components():
    from swarm_rescue.solutions.my_drone_eval import (
        path_components as eval_path_components,
    )

    return eval_path_components()


class PathFollowingDrone(RescueDrone):
    """Debug-only drone: isolate control from noisy pose measurements."""

    def __init__(self, **kwargs):
        super().__init__(**path_components(), **kwargs)
        self.trajectory = []
        self.path_errors = []
        self.arrival_tick = None

    def get_GPS(self):
        return np.asarray(self.true_position(), dtype=float)

    def get_yaw(self):
        return float(self.true_angle())

    def get_velocity(self):
        return np.asarray(self.true_velocity(), dtype=float)

    def control(self):
        commands = super().control()
        position = self.estimator.position_
        self.trajectory.append(tuple(position))

        starts = self.path_follower.positions[:-1]
        segments = np.diff(self.path_follower.positions, axis=0)
        fractions = np.clip(
            np.sum((position - starts) * segments, axis=1)
            / np.sum(segments * segments, axis=1), 0.0, 1.0,
        )
        projections = starts + fractions[:, None] * segments
        error = float(np.min(np.linalg.norm(projections - position, axis=1)))
        self.path_errors.append(error)

        final_error = np.linalg.norm(self.path_follower.positions[-1] - position)
        speed = np.linalg.norm(self.estimator.velocity_)
        on_final_segment = self.path_follower.index == len(PATH) - 2
        if on_final_segment and final_error < 5.0 and speed < 0.1:
            if self.arrival_tick is None:
                self.arrival_tick = self.elapsed_timestep
                print(f"Reached final target at tick {self.arrival_tick}")
        if self.elapsed_timestep % 150 == 0:
            print(
                f"tick={self.elapsed_timestep} "
                f"segment={self.path_follower.index + 1}/{len(PATH) - 1} "
                f"path_error={error:.2f}px final_error={final_error:.2f}px "
                f"speed={speed:.2f}px/tick health={self.drone_health}"
            )
        return commands

    def draw_top_layer(self):
        super().draw_top_layer()
        arcade.draw_line_strip(
            self.path_follower.positions.tolist(), (255, 165, 0), 2,
        )
        if len(self.trajectory) > 1:
            arcade.draw_line_strip(self.trajectory, (0, 200, 255), 2)
        for x, y in self.path_follower.positions[1:-1]:
            arcade.draw_circle_filled(float(x), float(y), 5, (255, 165, 0))
            arcade.draw_circle_outline(float(x), float(y), 5, (0, 0, 0), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    the_map = MapIntermediate01(
        drone_type=PathFollowingDrone, zones_config=(),
        use_shaders=False, seed=args.seed,
    )
    print("Orange: supplied path. Cyan: flown trajectory. Magenta: final target.")
    print("Debug experiment: true position, heading and velocity; no shaders.")
    gui = GuiSR(the_map=the_map, headless=args.headless, use_mouse_measure=True)
    try:
        gui.run()
    finally:
        drone = the_map.drones[0]
        if drone.path_errors:
            errors = np.asarray(drone.path_errors)
            print(
                f"Path error: RMS={np.sqrt(np.mean(errors**2)):.2f}px, "
                f"max={np.max(errors):.2f}px; "
                f"arrival_tick={drone.arrival_tick}; health={drone.drone_health}"
            )


if __name__ == "__main__":
    main()
