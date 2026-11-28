from collections.abc import Callable
from dataclasses import dataclass

from . import config
from .control.interface import Control
from .control.pd_controller import PDController
from .localisation.estimator import MeasuredState
from .localisation.interface import Localisation
from .mission.exploration import FrontierExplorer
from .mission.interface import Mission
from .mission.manager import MissionManager
from .navigation.astar import AStarPlanner as DefaultPlanner
from .navigation.interface import Navigation, PathPlanner
from .navigation.navigator import Navigator
from .navigation.path_follower import PathFollower
from .navigation.waypoints import WaypointProfile
from .visualization.debug_draw import DebugDraw
from .visualization.interface import Visualization
from .world.interface import World
from .world.lidar_mapper import LidarMapper
from .world.model import WorldModel
from .world.semantic_tracker import SemanticTracker
## example to test swapability
# comment out to have the actual path planner or change the naming
# this test planner just creates straight paths
#from .navigation.straight_line import StraightLinePlanner as DefaultPlanner


@dataclass(frozen=True)
class DroneComponents:
    localisation: Localisation
    world_model: World
    mission: Mission
    navigator: Navigation
    controller: Control
    visualization: Visualization | None


ComponentFactory = Callable[[tuple[int, int] | None], DroneComponents]


def build_components(
    map_size: tuple[int, int] | None, *, planner: PathPlanner | None = None,
    mission: Mission | None = None, debug_plot: bool = True, heading_goal: float | None = None,
) -> DroneComponents:
    
    """Build fresh per-drone components after map dimensions are available."""

    ## Planner ##
    if planner is None:
        planner = DefaultPlanner()


    ## Mission Type and Explorer##
    if mission is None:
        ## Explorer ##
        explorer = FrontierExplorer(config.MINIMUM_CLUSTER_SIZE, config.MINIMUM_DISTANCE_PX)
        
        ## Mission Type ##
        mission = MissionManager(
            explorer,
            frontier_arrival_px=config.FRONTIER_ARRIVAL_PX,
            return_approach=config.RETURN_APPROACH,
            release_distance_px=config.RELEASE_DISTANCE_PX,
            return_facing_distance_px=config.RETURN_FACING_DISTANCE_PX,
        )

    ## Localiser ##
    localisation = MeasuredState()

    ## Mapper ##
    mapper = LidarMapper(
        map_size=map_size if map_size is not None else (2000, 2000),
        frontier_update_interval=config.FRONTIER_UPDATE_INTERVAL,
    )

    ## Semantic ##
    tracker = SemanticTracker()

    ## World ##
    world_model = WorldModel(mapper, tracker)

    ## Path follower ##
    path_follower = PathFollower()

    ## Waypoints structure ##
    waypoint_profile = WaypointProfile(
        min_speed=config.MIN_PATH_SPEED,
        max_speed=config.MAX_PATH_SPEED,
    )

    ## Navigator ##
    navigator = Navigator(planner, path_follower, waypoint_profile, clearance_px=config.CLEARANCE_PX)

    ## Controller ##
    controller = PDController(heading_goal=heading_goal)
    
    ## Plotting (debug) ##
    visualization = DebugDraw() if debug_plot else None


    ## FINISH ##
    return DroneComponents(
        localisation=localisation,
        world_model=world_model,
        mission=mission,
        navigator=navigator,
        controller=controller,
        visualization=visualization,
    )
