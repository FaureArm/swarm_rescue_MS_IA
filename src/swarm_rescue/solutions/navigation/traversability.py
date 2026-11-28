import numpy as np
from skimage.draw import line
from skimage.morphology import isotropic_dilation

from ..world.grid import OccupancyGrid


# this function remains the same for everyone, so
# we keep it outside of path algo for now
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


def segment_is_traversable(grid, mask, start, goal) -> bool:
    x, y, valid = grid.world_to_grid(*start)
    end_x, end_y, end_valid = grid.world_to_grid(*goal)
    if not (valid and end_valid):
        return False
    xs, ys = line(int(x), int(y), int(end_x), int(end_y))
    # A diagonal must not squeeze between blocked orthogonal neighbors.
    return bool(
        mask[xs, ys].all()
        and mask[xs[1:], ys[:-1]].all()
        and mask[xs[:-1], ys[1:]].all()
    )
