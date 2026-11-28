import numpy as np

from ..data_types import Detection, ObjectKind, StateEstimate


class SemanticTracker:
    def __init__(self):
        self.rescue_center = None

    def update(self, pose: StateEstimate, detections: tuple[Detection, ...]):
        wounded = []
        for detection in detections:
            angle = pose.yaw + detection.angle
            point = pose.position + detection.distance * np.array([np.cos(angle), np.sin(angle)])
            if detection.kind is ObjectKind.WOUNDED:
                wounded.append(point)
            elif detection.kind is ObjectKind.RESCUE_CENTER and self.rescue_center is None:
                self.rescue_center = point
        return tuple(wounded), self.rescue_center
