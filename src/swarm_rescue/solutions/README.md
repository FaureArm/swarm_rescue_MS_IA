# Drone solutions

The evaluation drone selects exploration waypoints from frontier clusters and
flies towards them using a PD controller. Mapping is enabled by default, and
each drone has its own components, state and occupancy grid. The manual
waypoint mission remains available for experiments.

## Files

| File | What it contains / does |
| --- | --- |
| `my_drone_eval.py` | Configures the evaluation drone's exploration mission and mapper using experiment constants. `exploration_components()` shares one mapper between drone and mission; `waypoint_components(mapping=False)` retains flight without mapping. |
| `drone.py` | Contains `RescueDrone`: updates mapping, measured state, mission and control each simulation step, and returns movement and grasper commands. |
| `localisation.py` | Contains `MeasuredState`: stores measured position, heading and velocity. |
| `mission_planner.py` | Contains `FrontierExplorationMission`, which reselects a goal on frontier refresh or arrival, and the manual `WaypointMission`. Both return a `MissionDecision` containing destination and grasper intent. |
| `data_types.py` | Contains `OccupancyGrid` with bounded evidence, observation history, four-state classification and a frontier-cell mask; also `MissionDecision` and the currently unused `Waypoint` type. |
| `control.py` | Contains the PD waypoint controller: calculates forward/lateral thrust and rotation from the target, measured position, velocity and heading. Limits thrust and holds the initial heading unless another is supplied. |
| `mapping.py` | Contains `LidarMapper`: updates occupied pixels and the occupancy grid, caches frontiers, groups clusters and selects free-cell candidate waypoints. Current experiments use ground-truth pose. |
| `debug_draw.py` | Draws the goal and map pixels in the simulator; the Matplotlib occupancy plot shows frontier cells, all eligible per-cluster candidates and the chosen goal. |
| `my_drone_map_building.py` | Keeps the existing `MyDroneMapper` class name for scripts that use it; creates a `RescueDrone` with mapping enabled. |
| `navigation.py` | Empty for now. Will contain route planning and path following to turn a mission destination into the next motion target. |
| `swarm_coordination.py` | Empty for now. Will use the communicator to exchange peer data and propose assignments for the mission to accept. |

Flight currently goes straight towards the exploration destination. Route
planning, obstacle clearance, coordinated exploration and retrieval behavior
remain future work; the current mission keeps the grasper open.

`tests/test_rescue_drone.py` compares the refactor with the original commands and
map pixels, checks independent drone state, and runs short simulation tests.
`tests/fixtures/waypoint_baseline/` holds the original code used for comparison.

## Roadmap

1. **Probabilistic occupancy grid:** Replace the current permanent set of occupied pixels with an evolving representation of occupied, free, and unknown space.

   -> **DONE:** Pixel-size cells with bounded log odds, observation timestamps, bounds filtering, and separate free, occupied, unknown, and uncertain labels.

2. **Obstacle inflation:** Account for the drone's physical dimensions and safety margins.

   -> **DEFERRED:** Clearance representation and update strategy remain undecided.

3. **Grid collision checking:** Determine whether a planned path intersects occupied or unsafe space.
4. **Path planning:** Initially implement A*, with the possibility of experimenting with Theta*, D* Lite, or other planners.
5. **Path following:** Allow the drone to follow a path without stopping at every intermediate waypoint. Eventually support reference velocity, heading, and trajectory tracking.
6. **Online replanning:** Update planned routes as new obstacles are discovered or better routes become available.
7. **Frontier-based exploration:** Automatically select informative destinations for exploration.

   -> **PROTOTYPE DONE:** Cached frontier masks, connected clusters, and distance-based waypoint selection drive exploration; candidates and the selected goal are visualized, while obstacle-aware trajectories remain pending.

8. **Swarm coordination:** Enable drones to exchange map information, coordinate exploration, and distribute rescue objectives.

   -> **TODO:** Each drone currently keeps its own map; map sharing and coordinated assignments are not implemented.

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
