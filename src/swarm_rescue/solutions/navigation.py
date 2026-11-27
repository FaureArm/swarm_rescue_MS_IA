from swarm_rescue.solutions.Astar import AstartPlanner
from swarm_rescue.solutions.path_follower import PathFollower



class Navigator:
    def __init__(self, path_planner: AstartPlanner, path_follower: PathFollower):
        self.path_planner = path_planner
        self.path_follower = path_follower

    def clear(self) -> None:
        self.path_follower.has_path = False

    def set_goal(self, grid, start, goal) -> bool:
        self.path_follower.has_path = False
        if start is None or goal is None:
            return False

        route = self.path_planner.plan_path(grid, start, goal)
        if route is None or len(route) < 2:
            return False

        waypoints = self.path_planner.extract_waypoints(route)
        self.path_follower.set_path(
            waypoints, lookahead=15.0, gain=0.05, max_pos=5.0, max_vel=0.1,
        )
        return True

    def update(self, estimate):
        if not self.path_follower.has_path or estimate.position_ is None:
            return None
        return self.path_follower.update(estimate)