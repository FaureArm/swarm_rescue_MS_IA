"""Compare the component wiring with frozen, pre-refactor flight algorithms."""

import numpy as np
import pytest

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.true_state_guard import TrueStateAccessError
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.mapping import LidarMapper
from swarm_rescue.solutions.data_types import MissionDecision
from swarm_rescue.solutions.my_drone_basic_control import MyDroneBasicControl
from swarm_rescue.solutions.my_drone_eval import MyDroneEval, waypoint_components
from swarm_rescue.solutions.my_drone_map_building import MyDroneMapper
from tests.fixtures.waypoint_baseline.basic import MyDroneBasicControl as OldBasic
from tests.fixtures.waypoint_baseline.mapper import MyDroneMapper as OldMapper


@pytest.fixture
def component_only(monkeypatch):
    """Isolate deterministic calculations; live constructors are tested below."""
    monkeypatch.setattr(DroneAbstract, "__init__", lambda self, **kwargs: None)


def sensor_inputs(monkeypatch, drone, *, position, yaw, velocity, ranges=None):
    """Record the sensor call order, including the intentionally distinct true pose."""
    calls = []
    values = {
        "true_position": np.array([20.25, -30.75]),
        "true_angle": -0.4,
        "lidar_values": ranges,
        "lidar_rays_angles": np.array([-0.4, 0.0, 0.8]),
        "measured_gps_position": np.asarray(position, dtype=float),
        "measured_compass_angle": yaw,
        "measured_velocity": np.asarray(velocity, dtype=float),
    }
    for name, value in values.items():
        def read(name=name, value=value):
            calls.append(name)
            return value
        monkeypatch.setattr(drone, name, read)
    return calls


@pytest.mark.parametrize("mapping", [False, True])
@pytest.mark.parametrize("goal,position,yaw,velocity,heading_goal", [
    ((10, 15), (1, -2), 0.3, (0.2, -0.5), 0.4),
    ((1000, -2000), (-300, 150), 1.1, (10, -20), -2.0),  # saturation
    ((15, 20), (0, 0), np.pi - 0.01, (0, 0), -np.pi + 0.02),
    ((-20, 10), (0, 0), -np.pi + 0.01, (0, 0), np.pi - 0.02),
    ((10, 10), (0, 0), -1.2, (0.5, 1), None),  # latch the first heading
])
def test_commands_match_baseline(
    component_only, monkeypatch, mapping, goal, position, yaw, velocity, heading_goal
):
    old = (OldMapper if mapping else OldBasic)(goal=goal, heading_goal=heading_goal)
    new = RescueDrone(**waypoint_components(
        mapping=mapping, goal=goal, heading_goal=heading_goal
    ))
    for tick_yaw in (yaw, -yaw, 0.2):
        inputs = dict(position=position, yaw=tick_yaw, velocity=velocity,
                      ranges=np.array([0, 30, 275]))
        old_calls = sensor_inputs(monkeypatch, old, **inputs)
        new_calls = sensor_inputs(monkeypatch, new, **inputs)
        assert new.control() == old.control()
        assert new_calls == old_calls
        assert new.controller.heading_goal == old.heading_goal
        assert new.define_message_for_all() is None
        if mapping:
            assert new.mapper.map == old.map


@pytest.mark.parametrize("mapping,threshold", [(False, 0.5), (True, 1)])
@pytest.mark.parametrize("distance", [0, 0.49, 0.5, 0.75, 0.99, 1.0, 1.01])
def test_waypoint_arrival_uses_prior_tick(
    component_only, monkeypatch, mapping, threshold, distance
):
    draws = []
    def next_goal(low, high, size):
        draws.append((low, high, size))
        return np.array([42.0, -17.0])
    monkeypatch.setattr(np.random, "uniform", next_goal)
    old = (OldMapper if mapping else OldBasic)(goal=(0, 0))
    new = RescueDrone(**waypoint_components(mapping=mapping, goal=(0, 0)))

    for tick, position in enumerate(((distance, 0), (10, 10))):
        inputs = dict(position=position, yaw=0.2, velocity=(0.1, -0.3))
        sensor_inputs(monkeypatch, old, **inputs)
        sensor_inputs(monkeypatch, new, **inputs)
        assert new.control() == old.control()
        np.testing.assert_array_equal(new.mission.goal, old.goal)
        expected = [42, -17] if tick == 1 and distance < threshold else [0, 0]
        np.testing.assert_array_equal(new.mission.goal, expected)
    assert draws == [(-100, 100, 2)] * (2 if distance < threshold else 0)


@pytest.mark.parametrize("ranges", [
    None,
    np.array([]),
    np.array([np.nan, np.inf, -np.inf, -1, 0, 0.1, 1, 274.99, 275, 275.01, 300]),
    np.array([5, 5, 5, 5]),
])
@pytest.mark.parametrize("position,yaw", [((0.5, -0.5), 0), ((12.2, -14.7), 3.13)])
def test_lidar_pixels_match_baseline(component_only, monkeypatch, ranges, position, yaw):
    angles = np.linspace(-np.pi, np.pi, len(ranges)) if ranges is not None else None
    old = OldMapper()
    monkeypatch.setattr(old, "lidar_values", lambda: ranges)
    monkeypatch.setattr(old, "lidar_rays_angles", lambda: angles)
    new = LidarMapper()
    for _ in range(2):  # accumulation and duplicate handling
        old.update_lidar_map(np.array(position), yaw)
        new.update_lidar_map(np.array(position), yaw, ranges, angles)
        assert new.map == old.map


@pytest.mark.parametrize("mapping", [False, True])
def test_no_destination_preserves_short_circuit(component_only, monkeypatch, mapping):
    old = (OldMapper if mapping else OldBasic)()
    new = RescueDrone(**waypoint_components(mapping=mapping))
    old.goal = new.mission.goal = None
    inputs = dict(position=(0, 0), yaw=0, velocity=(0, 0), ranges=np.array([1, 2, 3]))
    old_calls = sensor_inputs(monkeypatch, old, **inputs)
    new_calls = sensor_inputs(monkeypatch, new, **inputs)
    assert new.control() == old.control()
    assert new_calls == old_calls
    assert "measured_gps_position" not in new_calls
    if mapping:
        assert new.mapper.map == old.map
        assert new.mapper.map


def test_mission_grasper_and_role_do_not_rebuild_components(component_only, monkeypatch):
    drone = MyDroneEval()
    original = (drone.estimator, drone.mapper, drone.controller)
    sensor_inputs(monkeypatch, drone, position=(1, 2), yaw=0.7, velocity=(0, 0),
                  ranges=np.array([1, 2, 3]))
    drone.control()
    pixels = drone.mapper.map.copy()
    drone.mission.role = "retrieval"
    # Stand in for a future policy; no retrieval algorithm is introduced.
    monkeypatch.setattr(drone.mission, "decide",
                        lambda previous: MissionDecision(np.array([20, 30]), 1))
    assert drone.control()["grasper"] == 1
    assert (drone.estimator, drone.mapper, drone.controller) == original
    assert drone.controller.heading_goal == 0.7
    assert drone.mapper.map == pixels


def test_component_state_is_independent(component_only, monkeypatch):
    first, second = MyDroneEval(), MyDroneEval()
    for attr in ("estimator", "mapper", "mission", "controller", "debug_draw"):
        assert getattr(first, attr) is not getattr(second, attr)
    sensor_inputs(monkeypatch, first, position=(1, 2), yaw=0.7, velocity=(3, 4),
                  ranges=np.array([1, 2, 3]))
    first.control()
    first.mission.goal[0] = 999
    first.mission.role = "survey"
    first.estimator.velocity_[0] = 99
    assert not second.mapper.map
    assert second.estimator.position_ is None
    np.testing.assert_array_equal(second.estimator.velocity_, [0, 0])
    np.testing.assert_array_equal(second.mission.goal, [150, 100])
    assert second.mission.role == "waypoint"
    assert second.controller.heading_goal is None

    shared_goal = np.array([1.0, 2.0])
    a = waypoint_components(goal=shared_goal)["mission"]
    b = waypoint_components(goal=shared_goal)["mission"]
    a.goal[0] = 100
    np.testing.assert_array_equal(b.goal, shared_goal)


@pytest.mark.parametrize("entry,mapping,goal", [
    (MyDroneEval, True, (150, 100)),
    (MyDroneBasicControl, False, (0, 0)),
    (MyDroneMapper, True, (0, 0)),
])
def test_entry_points_forward_simulator_keywords(entry, mapping, goal):
    misc = MiscData(size_area=(600, 400), number_drones=2)
    drone = entry(identifier=7, misc_data=misc, display_lidar_graph=False, name="showcase")
    assert isinstance(drone, RescueDrone)
    assert drone.identifier == 7
    assert drone.size_area == misc.size_area
    assert drone._misc_data is misc
    assert drone.name == "showcase"
    assert (drone.mapper is not None) == mapping
    np.testing.assert_array_equal(drone.mission.goal, goal)
    assert drone.define_message_for_all() is None


@pytest.fixture
def playground():
    pg = ClosedPlayground(size=(600, 400), seed=42, use_shaders=False)
    try:
        yield pg
    finally:
        pg.close_window()


@pytest.mark.parametrize("entry", [OldMapper, MyDroneEval])
def test_existing_mapping_guard_failure_is_preserved(playground, monkeypatch, entry):
    monkeypatch.delenv("SWARM_RESCUE_ALLOW_TRUE_STATE", raising=False)
    drone = entry(identifier=0)
    playground.add(drone, initial_coordinates=((0, 0), 0))
    with drone.participant_code(), pytest.raises(TrueStateAccessError, match="true_position"):
        drone.control()


@pytest.mark.parametrize("mapping", [False, True])
def test_short_live_smoke(playground, monkeypatch, mapping):
    # Only the mapping test uses the simulator's existing documented debug switch.
    # The preceding test verifies that its normal guard still rejects both versions.
    if mapping:
        monkeypatch.setenv("SWARM_RESCUE_ALLOW_TRUE_STATE", "1")
        drone = MyDroneEval(identifier=0)
    else:
        monkeypatch.delenv("SWARM_RESCUE_ALLOW_TRUE_STATE", raising=False)
        drone = RescueDrone(**waypoint_components(goal=(150, 100)), identifier=0)
    playground.add(drone, initial_coordinates=((0, 0), 0))
    playground.step()  # prime the actual sensors
    for _ in range(10):
        with drone.participant_code():
            commands = drone.control()
            assert drone.define_message_for_all() is None
        assert set(commands) == {"forward", "lateral", "rotation", "grasper"}
        assert all(np.isfinite(commands[key]) and abs(commands[key]) <= 1
                   for key in ("forward", "lateral", "rotation"))
        playground.step(all_commands={drone: commands})
    assert np.linalg.norm(drone.true_position()) > 0
    if mapping:
        assert drone.mapper.map
