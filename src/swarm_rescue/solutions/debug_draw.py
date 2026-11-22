import arcade
import matplotlib.pyplot as plt
import numpy as np


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

    def plot_occupancy_grid(self):
        """Create or refresh a separate, nonblocking occupancy-map window.

        Unknown cells are gray, free cells white, and occupied cells black.
        Axes show grid indices.
        """
        if self.mapper is None:
            return None

        # OccupancyGrid currently stores its values as [x, y]; imshow uses [y, x].
        grid = self.mapper.occupancy_grid_
        states = grid.classify_cells().T
        pixels = np.full(states.shape, 0.5, dtype=np.float32)
        pixels[states == grid.FREE] = 1.0
        pixels[states == grid.OCCUPIED] = 0.0
        height, width = pixels.shape
        extent = (-0.5, width - 0.5, -0.5, height - 0.5)

        if not hasattr(self, "_occupancy_plot") or not plt.fignum_exists(
            self._occupancy_plot[0].number
        ):
            plt.ion()
            fig, ax = plt.subplots()
            image = ax.imshow(
                pixels, cmap="gray", vmin=0, vmax=1, origin="lower",
                interpolation="nearest", extent=extent,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("Grid x (px)")
            ax.set_ylabel("Grid y (px)")
            ax.set_title("Occupancy grid (gray: unknown, white: free, black: occupied)")
            self._occupancy_plot = fig, ax, image
            plt.show(block=False)

        fig, ax, image = self._occupancy_plot
        image.set_data(pixels)
        image.set_extent(extent)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig

