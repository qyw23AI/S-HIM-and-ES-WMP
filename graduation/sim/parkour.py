"""A1 parkour lanes on the WMP heightfield and camera interfaces."""
import numpy as np

from wmp_lab.heightfield import SubTerrain
from wmp_lab.terrain import Terrain

from graduation.legacy.es_wmp.es_wmp_config import ESWMPA1ParkourCfg

KINDS = ESWMPA1ParkourCfg.terrain.column_types_override
FIRST_OBSTACLE_X = (2.15, 2.10, 2.20, 2.00)
SPAWN_X = .80


class ParkourTerrain(Terrain):
    """Four low-to-high difficulty lanes with a common forward spawn."""

    def __init__(self, cfg, num_robots):
        super().__init__(cfg, num_robots)
        local_x = np.array([1.4, 2.4, 3.2, 4.0, 4.8, 5.5, 6.2]) - SPAWN_X
        points = np.zeros((cfg.num_rows, cfg.num_cols, len(local_x), 2), dtype=np.float32)
        points[..., 0] = local_x
        for level in range(cfg.num_rows):
            amplitude = .08 + .20 * level / cfg.num_rows
            points[level, 3, 1:6, 1] = amplitude * np.array([1, -1, 1, -1, 1])
        self.waypoints = points

    def make_terrain(self, choice, difficulty, i=0, j=0):
        field = SubTerrain("parkour", width=self.width_per_env_pixels,
                           length=self.length_per_env_pixels, vertical_scale=self.cfg.vertical_scale,
                           horizontal_scale=self.cfg.horizontal_scale)
        raw = field.height_field_raw
        mid = self.width_per_env_pixels // 2

        def x(meters):
            return round(meters / field.horizontal_scale)

        def height(meters):
            return round(meters / field.vertical_scale)

        if j == 0:
            for start in (FIRST_OBSTACLE_X[0], 3.55, 4.95):
                raw[x(start):x(start + 0.12 + difficulty * 0.52)] = height(-0.65)
        elif j == 1:
            for step in range(5):
                start = FIRST_OBSTACLE_X[1] + step * 0.32
                raw[x(start):x(start + 0.32)] = height((min(step, 4 - step) + 1) * (.05 + .09 * difficulty))
        elif j == 2:
            for start in (FIRST_OBSTACLE_X[2], 3.5, 4.8):
                raw[x(start):x(start + 0.25)] = height(.07 + .22 * difficulty)
        else:
            raw[x(FIRST_OBSTACLE_X[3]):x(5.7)] = height(-0.55)
            for index, start in enumerate((FIRST_OBSTACLE_X[3], 2.8, 3.6, 4.4, 5.2)):
                y = mid + x((-1) ** index * (.08 + .20 * difficulty))
                half = x(.40 - .12 * difficulty)
                raw[x(start):x(start + .52 - .10 * difficulty), y - half:y + half] = height(.03 + .11 * difficulty)
        # Deterministic low-amplitude surface texture for depth-derived event synthesis.
        xx, yy = np.indices(raw.shape)
        texture = ((xx * 19 + yy * 11 + (xx * yy) % 7) % 3 - 1).astype(np.int16)
        raw[raw == 0] += texture[raw == 0]
        return field

    def add_terrain_to_map(self, terrain, row, col):
        super().add_terrain_to_map(terrain, row, col)
        self.env_origins[row, col] = [row * self.env_length + SPAWN_X, (col + 0.5) * self.env_width, 0.0]
