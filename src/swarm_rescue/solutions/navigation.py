import numpy as np
from skimage.morphology import isotropic_dilation

from swarm_rescue.solutions.data_types import OccupancyGrid


def build_traversable_mask(
    grid: OccupancyGrid, *, clearance_px: float = 15.0,
) -> np.ndarray:
    """Return a boolean [x, y] mask of free cells with obstacle clearance."""
    
    if not np.isfinite(clearance_px) or clearance_px < 0:
        raise ValueError("clearance_px must be finite and nonnegative")

    states = grid.classify_cells()
    occupied = states == grid.OCCUPIED

    # add the clearance to make sure the drone doesnt hit walls
    padded_obstacles = np.pad(occupied, 1, constant_values=True)
    # we use an already existing library for this, makes things simpler
    inflated_obstacles = isotropic_dilation(padded_obstacles, radius=clearance_px)
    clearance_zone = ~inflated_obstacles[1:-1, 1:-1] #remove padding and flip booleans

    # return cells that are free and touside clearance zone
    return (states == grid.FREE) & clearance_zone