# Swarm Rescue modular pipeline

Start with [drone.py](src/swarm_rescue/solutions/drone.py), then open
[assembly.py](src/swarm_rescue/solutions/assembly.py) to see which implementations
fly. The simulator still starts at `my_drone_eval.py`. Assembly creates a set of
components for each evaluation drone after the simulator supplies the map size.
Tune the evaluation in `config.py`.

## One control tick

```python
self.tick_observations = self.read_observations()
self.pose = parts.localisation.update(self)
self.world = parts.world_model.update(self)
self.intent = parts.mission.update(self)
self.navigation_result = parts.navigator.update(self)
commands = parts.controller.update(self)
```

The adapter passes the same drone to each component and publishes its return value
before calling the next one. The mission reads navigation feedback from the previous
tick. Visualization reads the finished tick after control.

| Package | Reads | Produces or owns |
| --- | --- | --- |
| `localisation/` | GPS, compass and velocity in `tick_observations` | Estimated pose and velocity, or `None` when a complete estimate is unavailable |
| `world/` | Pose, LiDAR and semantic detections | The occupancy grid and known objects. The world owns the grid and consumers borrow it for reading. |
| `mission/` | Pose, world, grasper sensor and previous navigation result | Destination, movement mode, facing and grasper intent. It owns the exploration and rescue transitions. |
| `navigation/` | Pose, grid, tick and mission intent | Route, progress or failure, and the motion reference. It owns the planner and path follower. |
| `control/` | Pose, intent and current motion reference | Simulator commands. Without a usable motion reference, it issues zero movement commands and applies the grasper intent. |
| `visualization/` | Published world, intent and navigation result | Map, frontier candidates, latest planning mask and current route |

Each package's `interface.py` declares the fields its `update()` may read, and the
value it must return. The contracts are structural Python protocols. They guide
type checking and let a test supply a small object with the same fields. Treat
published arrays as read-only. Shared records live in `data_types.py`. Positions are
world `[x, y]` pixels, velocity is px/tick, yaw is radians, and grid arrays use
`[x, y]` indexing.

A new exploration policy can use an existing public field such as
`world.rescue_center` without changing the world package. When a feature genuinely
needs new information, add it to the owning subsystem's published record and
declare it in the consumer's context. Keep ownership of that value in the producer.

## From goal to path to commands

1. The mission requests planned movement for exploration and return, or direct
   movement for the close approach to a wounded person.
2. For a new planned goal, the navigator builds the clearance mask and calls
   `planner.plan_path(grid, start, goal, traversable=mask)`. The returned route
   contains ordered world points including both endpoints. `None` means no route.
3. The navigator checks the route, prepares waypoints and speeds, and loads the
   follower once. It calls the follower on every subsequent tick until the route
   ends or a new one is requested. Failed return requests retry after ten ticks.
   Exploration can choose another candidate when the previous one is blocked.
4. The follower returns the current target and desired velocity. Control produces
   forward, lateral and rotation commands and applies the grasper intent. The plot
   reads the route and mask from `NavigationResult` for either planner.

The current return approach `(350, 130)` is still a scenario setting. The mission uses
the observed rescue center to decide when to release. The navigator retries
failed plans and replans on goal or grid replacement. Periodic revalidation of
an already active route against new obstacle evidence is future work. The
evaluation now uses the estimated pose for mapping.

## Try the planner swap

Run the deterministic swap test from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYGLET_AUDIO=silent MPLBACKEND=Agg \
  .venv/bin/python -m pytest -q tests/test_modular_pipeline.py
```

The test runs the normal `RescueDrone.control()` with a known grid and exercises
both planners. Both plan through clear space. A* detours around a wall.
`StraightLinePlanner` reports that goal as blocked. The tests also cover reusing a
route across ticks, selecting a new goal, pickup and release, and plotting the
demo planner's route.

For the evaluation, replace this one import in `assembly.py`:

```python
from .navigation.astar import AStarPlanner as DefaultPlanner
```

with:

```python
from .navigation.straight_line import StraightLinePlanner as DefaultPlanner
```

Alternatively, pass `planner=StraightLinePlanner()` to `build_components` in your
own scenario, as the test does. To add your own planner, implement `PathPlanner`
in `navigation/interface.py` in a new module and select it in assembly. A planner
can be tested with a grid, start and goal on their own.

Run the simulator with `.venv/bin/python -m swarm_rescue.launcher`. The right-hand
plot shows the latest mask and any planned route. The straight-line planner is
an integration demo. It will report blocked whenever a detour is needed.
