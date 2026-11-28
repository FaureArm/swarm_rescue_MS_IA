# How Swarm Rescue is wired

Start with [assembly.py](src/swarm_rescue/solutions/assembly.py) if you want to
change **which implementation runs**, [drone.py](src/swarm_rescue/solutions/drone.py)
if you want to see **what happens on each tick**, and a package's `interface.py`
if you want to know **what your implementation has to provide**.

## Where the components come from

1. [MyDroneEval](src/swarm_rescue/solutions/my_drone_eval.py) gives
   `RescueDrone` the function `build_components`.
2. `RescueDrone` calls that function once it has the map size. The factory
   creates and connects the component objects for that drone. It runs again on
   reset.
3. The factory returns a `DroneComponents` bundle. The drone stores it as
   `self.parts` and uses those instances on each tick.

The packages organize the code. The factory instantiates classes from them.
Some classes go directly into `DroneComponents`, such as the controller.
Others are passed to the component that uses them. The explorer goes into
`MissionManager`, the planner and follower go into `Navigator`, and the mapper
and semantic tracker go into `WorldModel`. Change the relevant constructor in
`build_components()` to select a different implementation. Use
[config.py](src/swarm_rescue/solutions/config.py) to tune the existing choices.

## What changed in the refactor

The comparison below is with `feature/path_planning`, which already had A*,
Kalman filtering, LiDAR mapping, and path following. The refactor introduced
shared contracts and changed the component code that implements them.

| Area | Before | After |
| --- | --- | --- |
| Startup and control loop | `my_drone_eval.py` assembled the concrete classes. `RescueDrone` passed the estimator, mapper, explorer, follower, and navigator into mission logic and called individual mapping methods. | `assembly.py` owns construction and `config.py` holds evaluation settings. `RescueDrone` reads the sensors and calls the top-level components in order, publishing each result for the next step. The subsystem protocols live in `interface.py` files and shared records live in `data_types.py`. |
| Localisation | `MeasuredState.update_estimate()` stored position, yaw and velocity on the estimator. Other components read those fields. | The Kalman filter moved into `localisation/` with its algorithm intact. `MeasuredState.update()` reads the tick observations and publishes a `StateEstimate`, or `None` when a usable estimate is unavailable. |
| Mapping and perception | The drone called the LiDAR mapper with simulator ground truth and updated semantic detections separately. | `WorldModel` calls the existing LiDAR grid update with estimated pose and combines the grid, frontiers and tracked objects into a `WorldView`. The map stays available when pose is missing, and that tick skips integrating measurements. |
| Exploration and mission | Mission classes could select goals, load routes and call the follower. The explorer read from a concrete mapper. | `FrontierExplorer` proposes candidates from a `WorldView`. `MissionManager` picks a goal and publishes a `MissionIntent` with movement mode, facing and grasper intent. It uses navigation feedback to reject a blocked frontier until the frontier revision changes. |
| A* and route preparation | The A* class built the clearance mask, searched the grid, and converted its result to speed-tagged waypoints. | A* still uses the same grid search library. `Navigator` now builds the mask and checks the returned route. `WaypointProfile` samples it and assigns speeds for either planner. It keeps the original path points if sampling would cut a blocked corner. |
| Path following | Missions could load waypoints or update the follower as part of their own decisions. | `Navigator` owns the active route and retries. It loads a successful route once through `Tracking.load()` and calls `track(pose)` on subsequent ticks. The follower retains its lookahead and speed tracking logic. |
| Flight control | The PD controller took the mission decision and read state from the estimator object. | It now reads the current pose, motion reference and grasper intent. Its core thrust calculations and gains are retained. Facing comes from explicit intent, and a missing pose or motion reference produces zero movement while applying the grasper intent. |
| Debug plot | The plot read the concrete planner's route and clearance settings. | It reads the published world, intent and `NavigationResult`. The same plot works with A* and the straight-line planner. |

The old exploration and mission experiments, including the work in progress
from before the refactor, are in [archive/legacy_solutions](archive/legacy_solutions).
The active solution uses the packaged implementations directly.

## What an interface does

An interface says which methods an implementation must provide, what the caller
supplies, and what the implementation returns. In
[navigation/interface.py](src/swarm_rescue/solutions/navigation/interface.py),
`PathPlanner` says that `Navigator` supplies a grid, start, goal and traversability
mask. The planner returns an ordered route in world `[x, y]` pixels, including
both endpoints, or `None` if no route exists.

The `...` in a `Protocol` means the implementation goes in a concrete class,
such as `AStarPlanner`. A class satisfies the protocol by providing the declared
methods with compatible signatures. The IDE or a type checker can flag a
mismatch. At runtime, Python calls the object that assembly supplied. Tests
check the meaning of the contract, such as whether the route starts and ends
at the requested points.

A **context** protocol describes the caller from the other side. When the drone
calls `parts.navigator.update(self)`, `self` is the actual drone.
`NavigationContext` lists the fields navigation relies on, including the pose,
world, mission intent, and tick observations. Inside navigation, the route
planner receives the inputs to `plan_path()` directly, so you can test it with
a small grid on its own.

Shared records such as `StateEstimate`, `WorldView`, and `MissionIntent` live
in [data_types.py](src/swarm_rescue/solutions/data_types.py). Positions use world
`[x, y]` pixels, velocity uses pixels per tick, yaw uses radians, and grid arrays
are indexed as `[x, y]`. Treat shared grids and arrays as read-only. If a feature
needs new data, add it to the owning subsystem's published output and document
it in the consumer's interface.

## Follow one control tick

| Step | What it produces |
| --- | --- |
| Read observations | `tick_observations` holds the measurements for this tick. |
| Localisation | `pose` holds estimated position, yaw and velocity, or `None` if a complete estimate is unavailable. |
| World model | `world` holds the occupancy grid, frontiers, and observed objects. Mapping uses the estimated pose. |
| Mission | `intent` holds the destination, movement mode, and grasper intent. It can use navigation feedback from the previous tick. |
| Navigation | `navigation_result` contains status, a motion reference when available, and route data for the plot. |
| Control | The simulator's forward, lateral, rotation and grasper commands. Without a motion reference, movement commands are zero and the grasper intent still applies. |
| Visualization | An optional debug plot of the map, goal, and navigation result. |

For example, the explorer proposes a frontier, the mission selects it as a
destination, and the navigator requests a route. The navigator prepares
waypoints and target speeds, loads the follower once, and calls it on each
following tick. The follower provides the target and desired velocity that
control uses. Close approach to a wounded person can request direct movement
instead. A failed route reports `BLOCKED`, so exploration can try another
candidate.

The return mission still plans toward the configured approach point `(350, 130)`
and uses the observed rescue center to decide when to release. Navigation
replans on a changed goal or replacement grid and retries failed requests.
An active route is not currently revalidated against new obstacle evidence.

## Try swapping an implementation

**Path planner.** In `assembly.py`, replace the import that defines
`DefaultPlanner` with this one. The factory still passes `DefaultPlanner()`
to the same navigator.

```python
from .navigation.straight_line import StraightLinePlanner as DefaultPlanner
```

Use `from .navigation.astar import AStarPlanner as DefaultPlanner` to switch back.
Both planners work through clear space. A* can detour around a wall, whereas
the straight-line demo reports blocked when the direct segment is obstructed.
For a one-off scenario, pass `planner=StraightLinePlanner()` to the factory.

**Explorer.** Import your explorer and replace `explorer = FrontierExplorer(...)`
inside `build_components()`. Implement
[`ExplorationPolicy.candidates(world, drone_position)`](src/swarm_rescue/solutions/mission/interface.py),
returning an `(N, 2)` NumPy array of world-coordinate destinations. Return an empty
`(0, 2)` array when there are no candidates. The mission currently picks the
nearest eligible candidate. If your explorer wants to nominate one destination,
return it as a `(1, 2)` array.

From the repository root, run the simulator:

```bash
.venv/bin/python -m swarm_rescue.launcher
```

Restore A* as the default before running the focused tests. The tests exercise
both planners through the same control loop and verify the clear-space and
wall cases.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYGLET_AUDIO=silent MPLBACKEND=Agg \
  .venv/bin/python -m pytest -q tests/test_modular_pipeline.py
```
