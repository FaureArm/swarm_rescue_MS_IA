import arcade
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


from .interface import DebugContext


class DebugDraw:
    def __init__(self, *, interval: int = 10):
        self._interval = interval

    def update(self, drone: DebugContext) -> None:
        if drone.tick_observations.timestep % self._interval == 0:
            self.plot_occupancy_grid(drone)

    def plot_occupancy_grid(self, drone: DebugContext):
        """Show the occupancy grid and the latest planning mask.

        Unknown cells are gray, uncertain cells light gray, free cells white,
        and occupied cells black. Cached frontier cells are green.
        Orange circles mark candidate waypoints; a magenta star marks the goal.
        Axes show grid indices.
        """

        # OccupancyGrid currently stores its values as [x, y]; imshow uses [y, x].
        grid = drone.world.grid
        states = grid.classify_cells().T
        mask = drone.navigation_result.traversable
        mask_pixels = np.zeros_like(states)
        mask_pixels[states == grid.FREE] = 1
        if mask is not None:
            mask_pixels[mask.T] = 2
        height, width = states.shape
        pixels = np.full((height, width, 3), 0.5, dtype=np.float32)
        pixels[states == grid.UNCERTAIN] = 0.75
        pixels[states == grid.FREE] = 1.0
        pixels[states == grid.OCCUPIED] = 0.0

        frontiers = drone.world.frontiers
        if frontiers is not None:
            # A cached frontier may have been observed since its last refresh.
            visible_frontiers = frontiers.T & (states == grid.UNKNOWN)
            pixels[visible_frontiers] = (0.0, 1.0, 0.0)

        extent = (-0.5, width - 0.5, -0.5, height - 0.5)

        if not hasattr(self, "_occupancy_plot") or not plt.fignum_exists(
            self._occupancy_plot[0].number
        ):
            plt.ion()
            fig, (ax, mask_ax) = plt.subplots(
                1, 2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained",
            )
            image = ax.imshow(
                pixels, origin="lower",
                interpolation="nearest", extent=extent,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("Grid x (px)")
            ax.set_ylabel("Grid y (px)")
            ax.set_title(
                "Occupancy grid (gray: unknown, light gray: uncertain,\n"
                "white: free, black: occupied, green: frontiers)"
            )
            candidates = ax.scatter(
                [], [], s=40, marker="o", facecolors="none",
                edgecolors="darkorange", linewidths=1.5,
                label="Candidate waypoints", zorder=3,
            )
            selected = ax.scatter(
                [], [], s=100, marker="*", color="magenta",
                edgecolors="black", linewidths=0.5,
                label="Selected waypoint", zorder=4,
            )
            ax.legend(loc="upper right")

            mask_image = mask_ax.imshow(
                mask_pixels, origin="lower", interpolation="nearest", extent=extent,
                cmap=ListedColormap(["#222222", "#adc4f0", "#ffffff"]),
                vmin=-0.5, vmax=2.5,
            )
            mask_ax.set_aspect("equal")
            mask_ax.set_xlabel("Grid x (px)")
            start_marker = mask_ax.scatter(
                [], [], c="deepskyblue", edgecolors="black", zorder=3, label="Drone estimate",
            )
            goal_marker = mask_ax.scatter(
                [], [], c="magenta", edgecolors="black", marker="*", s=100,
                zorder=3, label="Requested destination",
            )
            route_line, = mask_ax.plot(
                [], [], color="darkorange", linewidth=2, marker="o", markevery=[-1],
                markersize=5, label="Planned route",
            )
            mask_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3)
            self._occupancy_plot = (
                fig, ax, image, candidates, selected, mask_image, start_marker, goal_marker,
                route_line,
            )
            plt.show(block=False)

        (
            fig, ax, image, candidates, selected, mask_image, start_marker, goal_marker,
            route_line,
        ) = self._occupancy_plot
        image.set_data(pixels)
        image.set_extent(extent)
        mask_image.set_data(mask_pixels)
        mask_image.set_extent(extent)
        mask_image.axes.set_title(
            "Latest planning mask" if mask is not None else "Waiting for a route request"
        )
        start = None if drone.pose is None else drone.pose.position
        goal = drone.intent.destination
        for marker, point in ((start_marker, start), (goal_marker, goal)):
            offsets = np.empty((0, 2))
            if point is not None:
                x, y, in_bounds = grid.world_to_grid(*point)
                if in_bounds:
                    offsets = [[x, y]]
            marker.set_offsets(offsets)

        origin = np.array([grid.width // 2, grid.height // 2])
        route = drone.navigation_result.route
        if route is None:
            route_line.set_data([], [])
        else:
            route_line.set_data(*(route + origin).T)
        points = drone.intent.candidates
        candidates.set_offsets(np.empty((0, 2)) if points is None else points + origin)
        selected.set_offsets(
            np.empty((0, 2)) if goal is None
            else np.asarray(goal, dtype=float).reshape(1, 2) + origin
        )
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig

