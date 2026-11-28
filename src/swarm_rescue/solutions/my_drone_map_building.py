from functools import partial

from .assembly import build_components
from .drone import RescueDrone
from .mission.manager import FixedGoalMission


class MyDroneMapper(RescueDrone):
    def __init__(self, *, goal=None, heading_goal=None, **kwargs):
        factory = partial(
            build_components, mission=FixedGoalMission((0.0, 0.0) if goal is None else goal),
            heading_goal=heading_goal,
        )
        super().__init__(build_components=factory, **kwargs)
