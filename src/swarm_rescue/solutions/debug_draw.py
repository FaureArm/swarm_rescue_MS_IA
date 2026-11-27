import arcade
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class DebugDraw:
    def __init__(self, mission, mapper):
        self.mission = mission
        self.mapper = mapper

    @property
    def goal(self):
        return self.mission.goal

    @property
    def map(self):
        return self.mapper.lidar_map_ if self.mapper is not None else None

    def draw_top_layer(self):
        if self.map:
         arcade.draw_points(list(self.map), (0, 220, 255, 160), 2)
        if self.goal is None:
            return

        # draw a basic circle
        x, y = self.goal
        arcade.draw_circle_outline(
            center_x=float(x),
            center_y=float(y),
            radius=8,
            color=(255, 0, 255),
            border_width=2,
        )


    def plot_lidar_map(self):

        if not self.map:
            return

        if not hasattr(self, "_map_plot") or not plt.fignum_exists(
            self._map_plot[0].number
        ):
            plt.ion()
            fig, ax = plt.subplots()
            points = ax.scatter([], [], s=2, marker="s")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("World x (px)")
            ax.set_ylabel("World y (px)")
            self._map_plot = fig, ax, points
            plt.show(block=False)

        fig, ax, points = self._map_plot
        pixels = np.asarray(list(self.map))
        points.set_offsets(pixels)
        ax.update_datalim(pixels)
        ax.autoscale_view()
        ax.set_title(f"Occupied pixels: {len(self.map):,}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def plot_occupancy_grid(self, path_planner, *, start, goal):
        """Show the occupancy grid and the current A* traversability mask.

        Unknown cells are gray, uncertain cells light gray, free cells white,
        and occupied cells black. Cached frontier cells are green.
        Orange circles mark candidate waypoints; a magenta star marks the goal.
        Axes show grid indices.
        """
        if self.mapper is None:
            return None

        # OccupancyGrid currently stores its values as [x, y]; imshow uses [y, x].
        grid = self.mapper.occupancy_grid_
        states = grid.classify_cells().T
        traversable = path_planner.get_traversable_mask(grid).T
        mask_pixels = np.zeros_like(states)
        mask_pixels[states == grid.FREE] = 1
        mask_pixels[traversable] = 2
        height, width = states.shape
        pixels = np.full((height, width, 3), 0.5, dtype=np.float32)
        pixels[states == grid.UNCERTAIN] = 0.75
        pixels[states == grid.FREE] = 1.0
        pixels[states == grid.OCCUPIED] = 0.0

        frontiers = self.mapper.frontier_mask_
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
                zorder=3, label="Rescue center",
            )
            route_line, = mask_ax.plot(
                [], [], color="darkorange", linewidth=2, marker="o", markevery=[-1],
                markersize=5, label="A* route",
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
            f"A* mask ({path_planner.clearance_px:g} px clearance)\n"
            "White: traversable, blue: clearance buffer, dark: non-free"
        )
        for marker, point in ((start_marker, start), (goal_marker, goal)):
            offsets = np.empty((0, 2))
            if point is not None:
                x, y, in_bounds = grid.world_to_grid(*point)
                if in_bounds:
                    offsets = [[x, y]]
            marker.set_offsets(offsets)

        origin = np.array([grid.width // 2, grid.height // 2])
        if path_planner.current_route is None:
            route_line.set_data([], [])
        else:
            route_line.set_data(*(path_planner.current_route + origin).T)
        candidates.set_offsets(self.mapper.frontier_waypoints_ + origin)
        selected.set_offsets(
            np.empty((0, 2)) if self.goal is None
            else np.asarray(self.goal, dtype=float).reshape(1, 2) + origin
        )
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig

