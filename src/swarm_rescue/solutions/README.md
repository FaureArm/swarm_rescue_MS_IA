# Drone solutions

The drone flies towards a goal using a PD controller and chooses a new random
goal when it arrives. Mapping is enabled by default, starting at `(150.0, 100.0)`.
Each drone has its own components and state.

## Files

| File | What it contains / does |
| --- | --- |
| `my_drone_eval.py` | Chooses the drone used by the simulator. Creates its components, sets the starting goal, and enables mapping. `waypoint_components(mapping=False)` creates components for flight without mapping. |
| `drone.py` | Contains `RescueDrone`: reads sensors, calls the mapping, mission, state and control components each simulation step, and returns movement and grasper commands. |
| `localisation.py` | Contains `MeasuredState`: stores measured position, heading and velocity. |
| `mission_planner.py` | Contains `WaypointMission`: keeps the goal, checks arrival and chooses the next goal. Returns a `MissionDecision` containing the destination and grasper intent. |
| `data_types.py` | Contains shared data structures: `MissionDecision` (destination and grasper intent) and `Waypoint` (world coordinates `x`, `y`, in pixels; defined for future use). |
| `control.py` | Contains the PD waypoint controller: calculates forward/lateral thrust and rotation from the target, measured position, velocity and heading. Limits thrust and holds the initial heading unless another is supplied. |
| `mapping.py` | Contains `LidarMapper`: turns LiDAR hits into occupied map pixels and adds their immediate neighbours. Uses ground-truth pose for the current mapping experiments. |
| `debug_draw.py` | Draws the goal and map pixels in the simulator, and displays the map in Matplotlib. |
| `my_drone_map_building.py` | Keeps the existing `MyDroneMapper` class name for scripts that use it; creates a `RescueDrone` with mapping enabled. |
| `navigation.py` | Empty for now. Will contain route planning and path following to turn a mission destination into the next motion target. |
| `swarm_coordination.py` | Empty for now. Will use the communicator to exchange peer data and propose assignments for the mission to accept. |

Flight currently goes straight towards the destination. Future exploration,
survey and retrieval behaviours will choose destinations and grasper intent
through the mission, while keeping the drone's state, map and controller.

`tests/test_rescue_drone.py` compares the refactor with the original commands and
map pixels, checks independent drone state, and runs short simulation tests.
`tests/fixtures/waypoint_baseline/` holds the original code used for comparison.

## Roadmap

1. **Probabilistic occupancy grid:** Replace the current permanent set of occupied pixels with an evolving representation of occupied, free, and unknown space.
2. **Obstacle inflation:** Account for the drone's physical dimensions and safety margins.
3. **Grid collision checking:** Determine whether a planned path intersects occupied or unsafe space.
4. **Path planning:** Initially implement A*, with the possibility of experimenting with Theta*, D* Lite, or other planners.
5. **Path following:** Allow the drone to follow a path without stopping at every intermediate waypoint. Eventually support reference velocity, heading, and trajectory tracking.
6. **Online replanning:** Update planned routes as new obstacles are discovered or better routes become available.
7. **Frontier-based exploration:** Automatically select informative destinations for exploration.
8. **Swarm coordination:** Enable drones to exchange map information, coordinate exploration, and distribute rescue objectives.

## Run

Run these commands from the repository root using the existing `.venv`.
`SWARM_RESCUE_USE_SHADERS` selects GPU shaders (`1`) or CPU sensor calculations
(`0`). `SWARM_RESCUE_ALLOW_TRUE_STATE` permits (`1`) or blocks (`0`) ground-truth
reads during control; measured sensors remain noisy in either case.

The current mapper needs ground-truth access. The commands marked
"ground-truth access blocked" raise `TrueStateAccessError` when mapping runs.
Flight without a mapper can run with ground-truth access blocked.

```bash
# Shaders enabled, ground-truth access enabled (eval)
SWARM_RESCUE_USE_SHADERS=1 SWARM_RESCUE_ALLOW_TRUE_STATE=1 uv run -m swarm_rescue.launcher

# Shaders disabled, ground-truth access enabled (experiments)
SWARM_RESCUE_USE_SHADERS=0 SWARM_RESCUE_ALLOW_TRUE_STATE=1 uv run -m examples.test_basic_control

# Shaders enabled, ground-truth access blocked
SWARM_RESCUE_USE_SHADERS=1 SWARM_RESCUE_ALLOW_TRUE_STATE=0 uv run -m swarm_rescue.launcher
```

Append `--config config/<plan>.yml` to choose an evaluation plan, or `--headless`
to hide the window (a display/OpenGL context is still required).
