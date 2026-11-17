from swarm_rescue.solutions.my_drone_random import MyDroneRandom
from swarm_rescue.solutions.my_drone_lidar_communication import MyDroneLidarCommunication
from swarm_rescue.solutions.my_drone_motionless import MyDroneMotionless
from swarm_rescue.solutions.my_drone_basic_control import MyDroneBasicControl
from swarm_rescue.solutions.my_drone_map_building import MyDroneMapper
from swarm_rescue.solutions.my_drone_control_and_rescue import MyDroneControlAndRescue
from swarm_rescue.solutions.my_smart_drone import MySmartDrone


class MyDroneEval(MyDroneControlAndRescue):
    """
    Evaluation drone class that inherits from MyDroneRandom.

    This class can be extended to implement custom evaluation logic.
    """

    def __init__(self, **kwargs):
        super().__init__(goal=(150.0,100.0), **kwargs)


