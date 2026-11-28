from .assembly import build_components
from .drone import RescueDrone


class MyDroneEval(RescueDrone):
    def __init__(self, **kwargs):
        super().__init__(build_components=build_components, **kwargs)
