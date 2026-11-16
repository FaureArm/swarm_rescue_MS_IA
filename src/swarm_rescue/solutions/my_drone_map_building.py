from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.my_drone_eval import waypoint_components


class MyDroneMapper(RescueDrone):
    def __init__(self, *, goal=None, heading_goal=None, **kwargs):
        super().__init__(
            **waypoint_components(
                mapping=True, goal=goal, heading_goal=heading_goal
            ),
            **kwargs,
        )
