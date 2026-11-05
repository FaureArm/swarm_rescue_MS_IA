from swarm_rescue.solutions.my_drone_random import MyDroneRandom
from swarm_rescue.solutions.my_drone_lidar_communication import MyDroneLidarCommunication
from swarm_rescue.solutions.my_drone_motionless import MyDroneMotionless
from swarm_rescue.solutions.my_drone_basic_control import MyDroneBasicControl


class MyDroneEval(MyDroneBasicControl):
    """
    Evaluation drone class that inherits from MyDroneRandom.

    This class can be extended to implement custom evaluation logic.
    """

    def __init__(self, **kwargs):
        super().__init__(goal=(150.0, 100.0), **kwargs)


