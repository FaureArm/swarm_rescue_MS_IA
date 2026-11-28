"""Exercise planner substitution through RescueDrone.control with a known map."""

from dataclasses import replace

import numpy as np
import pytest

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.solutions.assembly import build_components
from swarm_rescue.solutions.data_types import (
    Activity, Detection, MissionIntent, MovementMode, NavigationStatus, ObjectKind, Observations, StateEstimate, WorldView,
)
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.mission.manager import FixedGoalMission
from swarm_rescue.solutions.navigation.straight_line import StraightLinePlanner
from swarm_rescue.solutions.visualization.debug_draw import DebugDraw
from swarm_rescue.solutions.world.grid import OccupancyGrid


START = np.array([-60.0, 0.0])
GOAL = np.array([60.0, 0.0])
SIZE = (220, 180)


class KnownWorld:
    def __init__(self, *, wall=False):
        self.grid = OccupancyGrid(*SIZE)
        self.grid.log_odds.fill(self.grid.MIN_LOG_ODDS)
        self.grid.last_observed.fill(0)
        if wall:
            x, y, _ = self.grid.world_to_grid(0, np.arange(-35, 36))
            self.grid.log_odds[x, y] = self.grid.MAX_LOG_ODDS
        self.wounded = ()
        self.rescue_center = None

    def update(self, drone):
        return WorldView(self.grid, wounded=self.wounded, rescue_center=self.rescue_center)


class CountingStraightLinePlanner(StraightLinePlanner):
    def __init__(self):
        self.calls = 0

    def plan_path(self, grid, start, goal, *, traversable):
        self.calls += 1
        return super().plan_path(grid, start, goal, traversable=traversable)


@pytest.fixture
def flight(monkeypatch):
    def init(drone, **kwargs):
        drone.size_area = SIZE

    monkeypatch.setattr(DroneAbstract, "__init__", init)

    def launch(world, *, planner=None, mission=None, localisation=None, debug_plot=False):
        def factory(map_size):
            components = build_components(map_size, planner=planner, mission=mission, debug_plot=debug_plot)
            return replace(
                components, world_model=world,
                localisation=localisation if localisation is not None else components.localisation,
            )

        drone = RescueDrone(build_components=factory)

        def tick(timestep, *, position=START, holding=False, yaw=0.0):
            drone.read_observations = lambda: Observations(
                timestep, position, yaw, np.zeros(2), None, None,
                holding_wounded=holding,
            )
            return drone.control()

        return drone, tick

    return launch


@pytest.mark.parametrize("wall", [False, True])
def test_planner_swap_in_same_control_loop(flight, wall):
    world = KnownWorld(wall=wall)
    mission = FixedGoalMission(GOAL, MovementMode.PLANNED)
    astar, tick_astar = flight(world, mission=mission)
    straight, tick_straight = flight(world, planner=StraightLinePlanner(), mission=mission)
    commands = tick_astar(0)
    straight_commands = tick_straight(0)

    assert astar.navigation_result.status is NavigationStatus.FOLLOWING
    np.testing.assert_array_equal(astar.navigation_result.route[[0, -1]], [START, GOAL])
    assert commands["forward"] > 0
    if wall:
        assert abs(astar.navigation_result.route[:, 1]).max() > 35
        assert straight.navigation_result.status is NavigationStatus.BLOCKED
        assert straight.navigation_result.route is None
        assert straight_commands == dict(forward=0.0, lateral=0.0, rotation=0.0, grasper=0)
    else:
        assert straight.navigation_result.status is NavigationStatus.FOLLOWING
        np.testing.assert_array_equal(straight.navigation_result.route, [START, GOAL])
        assert straight_commands["forward"] > 0


def test_route_reused_then_replanned_on_new_goal(flight):
    planner = CountingStraightLinePlanner()
    class ChangingMission:
        goal = GOAL

        def update(self, drone):
            return MissionIntent(Activity.EXPLORING, self.goal, MovementMode.PLANNED)

    mission = ChangingMission()
    drone, tick = flight(KnownWorld(), mission=mission, planner=planner)
    tick(0)
    initial_route = drone.navigation_result.route
    tick(1, position=np.array([-50.0, 0.0]))
    assert drone.navigation_result.route is initial_route
    assert planner.calls == 1

    new_goal = np.array([30.0, 40.0])
    mission.goal = new_goal
    tick(2)
    assert planner.calls == 2
    np.testing.assert_array_equal(drone.navigation_result.route[-1], new_goal)


def test_pickup_blocked_return_and_release(flight):
    planner = CountingStraightLinePlanner()
    world = KnownWorld()
    world.wounded = (np.array([-52.0, 0.0]),)
    world.rescue_center = np.array([350.0, 130.0])
    class ExactPose:
        def update(self, drone):
            sample = drone.tick_observations
            if sample.yaw is None:
                return None
            return StateEstimate(sample.position, sample.yaw, sample.velocity)

    drone, tick = flight(world, planner=planner, localisation=ExactPose())

    assert tick(0)["grasper"] == 1
    assert drone.intent.activity is Activity.GRASPING_WOUNDED
    assert planner.calls == 0  # Close approach requests direct movement.

    commands = tick(1, holding=True)
    assert drone.intent.activity is Activity.SEARCHING_RESCUE_CENTER
    assert drone.navigation_result.status is NavigationStatus.BLOCKED
    assert commands == dict(forward=0.0, lateral=0.0, rotation=0.0, grasper=1)
    assert tick(2, holding=True)["grasper"] == 1
    assert planner.calls == 1

    assert tick(3, holding=True, yaw=None)["grasper"] == 1
    assert drone.pose is None
    assert tick(4, holding=True, position=world.rescue_center)["grasper"] == 0
    assert drone.intent.activity is Activity.DROPPING_AT_RESCUE_CENTER
    assert tick(5, position=world.rescue_center)["grasper"] == 0
    assert drone.intent.activity is Activity.STAND_BY


def test_real_world_and_estimator_handle_missing_and_valid_observations(monkeypatch):
    monkeypatch.setattr(DroneAbstract, "__init__", lambda drone, **kw: setattr(drone, "size_area", SIZE))
    drone = RescueDrone(build_components=lambda size: build_components(size, debug_plot=False))
    drone.read_observations = lambda: Observations(0, None, None, None, None, None)
    assert drone.control()["forward"] == 0.0
    assert drone.pose is None
    assert drone.world.frontiers is None

    drone.read_observations = lambda: Observations(
        1, START, 0.0, np.zeros(2), np.array([10.0]), np.array([0.0]),
        (Detection(ObjectKind.WOUNDED, 10.0, 0.0),),
    )
    assert drone.control()["grasper"] == 1
    x, y, _ = drone.world.grid.world_to_grid(-50.0, 0.0)
    assert drone.world.grid.classify_cells()[x, y] == OccupancyGrid.OCCUPIED
    np.testing.assert_allclose(drone.intent.destination, [-50.0, 0.0])


def test_visualization_consumes_results_from_straight_line_planner(flight, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    drone, tick = flight(
        KnownWorld(), planner=StraightLinePlanner(),
        mission=FixedGoalMission(GOAL, MovementMode.PLANNED),
    )
    tick(0)
    figure = DebugDraw().plot_occupancy_grid(drone)
    try:
        route_plot = figure.axes[1].lines[0]
        offset = np.array([drone.world.grid.width // 2, drone.world.grid.height // 2])
        np.testing.assert_array_equal(np.column_stack(route_plot.get_data()), [START + offset, GOAL + offset])
    finally:
        plt.close(figure)


def test_frontier_mission_requests_planned_movement(flight):
    world = KnownWorld()
    # Publish a long observed boundary to exercise real frontier clustering and selection.
    world.grid.last_observed[world.grid.width // 2:, :] = -np.inf
    world.grid.log_odds[world.grid.width // 2:, :] = 0.0

    class FrontierWorld(KnownWorld):
        def update(self, drone):
            return WorldView(self.grid, self.grid.frontier_mask(), 1)

    frontier_world = FrontierWorld()
    frontier_world.grid = world.grid
    planner = CountingStraightLinePlanner()
    drone, tick = flight(frontier_world, planner=planner)
    tick(0)

    assert drone.intent.mode is MovementMode.PLANNED
    assert drone.intent.destination is not None
    assert drone.intent.candidates.shape == (1, 2)
    assert planner.calls == 1
    assert drone.navigation_result.status is NavigationStatus.FOLLOWING
