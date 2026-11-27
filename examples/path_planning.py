"""Known-map A* development scenario, independent of exploration and Kalman.

Run: SWARM_RESCUE_USE_SHADERS=1 SWARM_RESCUE_ALLOW_TRUE_STATE=1 uv run --no-sync python -m examples.path_planning
Add --check for setup validation, or --grid-only to inspect navigation clearance.
Ground truth is deliberately used only in this debug scenario. No route is
supplied until plan_path() is implemented. No grasping/rescue mission runs.
"""

import argparse
import multiprocessing
from types import SimpleNamespace

import arcade
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from skimage.draw import line
from skimage.measure import label

from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.normal_wall import NormalWall
from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.solutions.Astar import AstartPlanner, build_traversable_mask
from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.data_types import (
    Activity, MissionDecision, MotionReference, OccupancyGrid, Waypoint,
)
from swarm_rescue.solutions.path_follower import PathFollower


GOAL = np.array([-310.0, -140.0])  # 40 px above the stationary person.
WAYPOINT_SPEED_COLORS = ListedColormap(colormaps["viridis"](np.linspace(0, 0.75, 12)))


def plan_path(grid: OccupancyGrid, start: np.ndarray, goal: np.ndarray):
    """A* insertion point: return world-coordinate (x, y) points, or None.

    Include start and goal. The grid stores raw obstacles without drone-radius
    inflation; traversability and clearance belong in the future planner.
    """
    return AstartPlanner().plan_path(grid, start, goal)


def build_known_grid(the_map):
    """Rasterize this scenario's physical walls/person, without LiDAR noise.

    All timestamps are zero (observed), including blocked exterior padding.
    Cells within half a pixel diagonal of a shape are conservatively occupied.
    Rescue/return areas are traversable; the drone itself is excluded.
    """
    grid = OccupancyGrid(*the_map.size_area)
    x = np.arange(grid.width) - grid.width // 2
    y = np.arange(grid.height) - grid.height // 2
    width, height = the_map.size_area
    interior = (np.abs(x[:, None]) < width / 2) & (
        np.abs(y[None, :]) < height / 2
    )
    grid.log_odds[:] = np.where(interior, grid.MIN_LOG_ODDS, grid.MAX_LOG_ODDS)
    grid.last_observed.fill(0.0)
    cell_radius = np.sqrt(0.5)
    for element in the_map.playground.elements:
        if not isinstance(element, (NormalWall, WoundedPerson)):
            continue
        for shape in element.pm_shapes:
            bounds = shape.cache_bb()
            for world_x in range(int(np.floor(bounds.left - cell_radius)),
                                 int(np.ceil(bounds.right + cell_radius)) + 1):
                for world_y in range(int(np.floor(bounds.bottom - cell_radius)),
                                     int(np.ceil(bounds.top + cell_radius)) + 1):
                    if shape.point_query((world_x, world_y)).distance <= cell_radius:
                        gx, gy, valid = grid.world_to_grid(world_x, world_y)
                        if valid:
                            grid.log_odds[gx, gy] = grid.MAX_LOG_ODDS
    return grid


class PathPlanningDrone(DroneAbstract):
    """Hold position without a route; use the existing follower when supplied."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = None
        self.goal = GOAL.copy()
        self.route = None
        self.path_follower = PathFollower()
        self.controller = PDController()
        self.trajectory = []
        self.arrived = False

    def define_message_for_all(self):
        return None

    def compute_observations(self):
        """Skip unused sensors in this known-map, true-state experiment.

        In particular, semantic UID decoding is unnecessary here and can fail
        on the shader backend. Production drones retain normal sensor updates.
        """
        return None

    def set_route(self, route):
        """Install the future planner output; start/goal are world coordinates."""
        points = np.asarray(route, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.all(np.isfinite(points))
                or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0)):
            raise ValueError("Route must contain at least two finite, distinct consecutive points")
        if not np.allclose(points[0], self.start) or not np.allclose(points[-1], self.goal):
            raise ValueError("Route must include the scenario start and goal")
        self.route = points.copy()
        self.arrived = False
        waypoints = AstartPlanner(clearance_px = 25, max_speed=10, min_speed=1).extract_waypoints(points)
        waypoints[-1] = Waypoint(*points[-1], 0.0, None)
        self.path_follower.set_path(waypoints, lookahead=15.0, gain=0.05,
                                    max_pos=5.0, max_vel=0.1)
        speeds = self.path_follower.speeds
        colors = WAYPOINT_SPEED_COLORS(speeds / (speeds.max() or 1.0), bytes=True)
        self.waypoint_color_groups = {}
        for position, color in zip(self.path_follower.positions.tolist(), colors.tolist()):
            self.waypoint_color_groups.setdefault(tuple(color), []).append(position)

    def control(self):
        estimate = SimpleNamespace(
            position_=np.asarray(self.true_position(), dtype=float),
            yaw_=float(self.true_angle()),
            velocity_=np.asarray(self.true_velocity(), dtype=float),
        )
        self.trajectory.append(tuple(estimate.position_))
        if self.route is None:
            motion = MotionReference(self.start, None, None)
        elif (np.linalg.norm(estimate.position_ - self.goal) < 5.0
              and np.linalg.norm(estimate.velocity_) < 0.1):
            if not self.arrived:
                print(f"Reached goal at tick {self.elapsed_timestep}")
            self.arrived = True
            motion = MotionReference(self.goal, None, None)
        elif self.arrived:
            motion = MotionReference(self.goal, None, None)
        else:
            motion = self.path_follower.update(estimate)
        return self.controller.control(
            MissionDecision(motion, 0, Activity.EXPLORING), estimate,
        )

    def draw_top_layer(self):
        arcade.draw_circle_outline(*self.goal, 8, (255, 0, 255), 2)
        if self.route is not None:
            arcade.draw_line_strip(self.route.tolist(), (255, 165, 0), 2)
        if len(self.trajectory) > 1:
            arcade.draw_line_strip(self.trajectory, (0, 200, 255), 2)
        if self.route is not None:
            for color, positions in self.waypoint_color_groups.items():
                arcade.draw_points(positions, color, 6)


def create_scenario(seed=42):
    the_map = MapIntermediate01(
        drone_type=PathPlanningDrone, zones_config=(), use_shaders=None, seed=seed,
    )
    for element in the_map.playground.elements:
        if isinstance(element, WoundedPerson):
            element.clear_path()
    drone = the_map.drones[0]
    drone.start = np.asarray(drone.true_position(), dtype=float).copy()
    drone.grid = build_known_grid(the_map)
    route = plan_path(drone.grid, drone.start.copy(), drone.goal.copy())
    if route is not None:
        drone.set_route(route)
    return the_map


def validate_scenario(drone):
    """Check setup, including a necessary detour and point-cell reachability."""
    grid = drone.grid
    states = grid.classify_cells()
    assert np.all(np.isfinite(grid.last_observed)), "Map must be fully observed"
    assert np.all((states == grid.FREE) | (states == grid.OCCUPIED))
    sx, sy, start_valid = grid.world_to_grid(*drone.start)
    gx, gy, goal_valid = grid.world_to_grid(*drone.goal)
    assert start_valid and goal_valid
    assert states[sx, sy] == grid.FREE and states[gx, gy] == grid.FREE
    xs, ys = line(int(sx), int(sy), int(gx), int(gy))
    assert np.any(states[xs, ys] == grid.OCCUPIED), "Direct route must cross a wall"
    regions = label(states == grid.FREE, connectivity=1)
    assert regions[sx, sy] == regions[gx, gy], "A point-cell detour must exist"
    print(f"Setup OK: grid={states.shape}, start={drone.start}, goal={drone.goal}")
    print("Fully observed; endpoints free; direct route blocked; detour exists.")


def plot_grid(drone, *, waypoint_positions=None, waypoint_speeds=None, block=True):
    import matplotlib.pyplot as plt

    grid = drone.grid
    known_free = grid.classify_cells() == grid.FREE
    traversable = build_traversable_mask(grid)
    cell_colors = np.where(traversable, 2, np.where(known_free, 1, 0))
    x_min = -(grid.width // 2) - 0.5
    y_min = -(grid.height // 2) - 0.5

    figure, axes = plt.subplots(layout="constrained")
    image = axes.imshow(
        cell_colors.T, origin="lower", interpolation="nearest",
        cmap=ListedColormap(["#222222", "#adc4f0", "#ffffff"]),
        vmin=-0.5, vmax=2.5,
        extent=(x_min, x_min + grid.width, y_min, y_min + grid.height),
    )
    colorbar = figure.colorbar(image, ax=axes, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["Non-free", "Clearance buffer", "Traversable"])
    axes.scatter(*drone.start, color="blue", label="Start", zorder=4)
    axes.scatter(*drone.goal, color="magenta", marker="*", label="Goal", zorder=4)
    if drone.route is not None:
        axes.plot(*drone.route.T, color="darkorange", label="Planned route")
    if waypoint_positions is not None and waypoint_speeds is not None:
        markers = axes.scatter(
            *waypoint_positions.T, c=waypoint_speeds, cmap=WAYPOINT_SPEED_COLORS,
            vmin=0, vmax=waypoint_speeds.max() or 1.0, s=20, zorder=3,
        )
        figure.colorbar(markers, ax=axes, orientation="horizontal",
                        label="Waypoint speed (px/tick)")
    axes.set(xlabel="World x (px)", ylabel="World y (px)",
             title="Occupancy grid and navigation clearance")
    axes.legend()
    plt.show(block=block)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--grid-only", action="store_true")
    mode.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    the_map = create_scenario(args.seed)
    try:
        drone = the_map.drones[0]
        validate_scenario(drone)
        if args.check:
            return
        waypoint_data = {}
        if drone.route is not None:
            waypoint_data = {
                "waypoint_positions": drone.path_follower.positions,
                "waypoint_speeds": drone.path_follower.speeds,
            }
        if args.grid_only:
            plot_grid(drone, **waypoint_data)
            return
        if drone.route is None:
            print("No planner yet: holding start. Implement plan_path() to supply a route.")
        print("Orange: planned route. Purple to green dots: slow to fast waypoints.")
        print("Cyan: flown trajectory. Magenta: goal.")
        print("Follower corner handling remains unvalidated for future A* routes.")
        gui = GuiSR(the_map=the_map, headless=args.headless, use_mouse_measure=True)
        plot_process = None
        if not args.headless:
            # Keep Matplotlib's GUI/event loop separate from Arcade's OpenGL
            # context. Pass only the static planning data, never the simulator.
            plot_data = SimpleNamespace(
                grid=drone.grid, start=drone.start, goal=drone.goal,
                route=drone.route,
            )
            plot_process = multiprocessing.get_context("spawn").Process(
                target=plot_grid, args=(plot_data,),
                kwargs=waypoint_data, daemon=True,
            )
            plot_process.start()
        try:
            gui.run()
        finally:
            if plot_process is not None:
                if plot_process.is_alive():
                    plot_process.terminate()
                plot_process.join()
            gui.close()
    finally:
        the_map.playground.close_window()


if __name__ == "__main__":
    main()
