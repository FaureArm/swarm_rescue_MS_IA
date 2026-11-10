from swarm_rescue.solutions.my_drone_random import MyDroneRandom
from swarm_rescue.solutions.my_drone_lidar_communication import MyDroneLidarCommunication
from swarm_rescue.solutions.my_drone_motionless import MyDroneMotionless
from swarm_rescue.solutions.my_drone_basic_control import MyDroneBasicControl
from swarm_rescue.solutions.my_drone_map_building import MyDroneMapper


class MyDroneEval(MyDroneMapper):
    """
    Evaluation drone class that inherits from MyDroneRandom.

    This class can be extended to implement custom evaluation logic.
    """

    def __init__(self, **kwargs):
        super().__init__(goal=(-310.0, -180.0), **kwargs)


