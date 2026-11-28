import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import heapq

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.constants import (
    FOV_SEMANTIC_SENSOR, MAX_RANGE_LIDAR_SENSOR, RESOLUTION_SEMANTIC_SENSOR,
)
from swarm_rescue.simulation.utils.true_state_guard import true_state_is_allowed
from swarm_rescue.simulation.utils.utils import normalize_angle


logger = logging.getLogger(__name__)
Point = Tuple[float, float]
Cell = Tuple[int, int]
Bounds = Tuple[float, float, float, float]


@dataclass
class Issue:
    """Une issue (ouverture) détectée sur le périmètre d'une pièce."""
    centre: Point
    outward: Point
    visited: bool = False


@dataclass
class PathNode:
    """Élément de la liste chaînée des points de parcours du drone."""
    entry: Point
    previous: Optional["PathNode"] = None
    next: Optional["PathNode"] = None
    incoming: Optional[Issue] = None
    centre: Optional[Point] = None
    issues: List[Issue] = field(default_factory=list)
    bounds: Optional[Bounds] = None
    limits: Tuple[Optional[float], Optional[float],
                  Optional[float], Optional[float]] = (None, None, None, None)
    visited: bool = False


class DroneBis(DroneAbstract):
    CELL = 10.0
    WAYPOINT_TOLERANCE = 30.0
    OBSTACLE_DISTANCE = 30.0
    SLOW_DISTANCE = 150.0
    MAX_SPEED = 6.0
    CROSSING_DISTANCE = 40.0
    SURVEY_INTERVAL = 10
    SURVEY_ROW_SPACING = 10
    MIN_WALL_CELLS = 8
    MIN_EXIT_CELLS = 4
    WALL_CLEAR_SCANS = 3
    WALL_CONFIRM_SCANS = 3
    SURVEY_STUCK_LIMIT = 5
    STALL_STEPS = 20
    STALL_DISTANCE = 5.0
    SEMANTIC_ANGLE_TOLERANCE = math.radians(FOV_SEMANTIC_SENSOR) / (
        RESOLUTION_SEMANTIC_SENSOR - 1
    )
    SEMANTIC_DISTANCE_TOLERANCE = 20.0

    def __init__(self, identifier=None, misc_data=None, **kw):
        super().__init__(identifier=identifier, misc_data=misc_data,
                         display_lidar_graph=False, **kw)
        self.estimated_position: Optional[Point] = None
        self.heading: Optional[float] = None
        self.map: Dict[Cell, int] = {}
        self.room_map: Dict[Cell, int] = {}
        self.wall_clear_scans: Dict[Cell, int] = {}
        self.wall_confirm_scans: Dict[Cell, int] = {}
        self.contact_walls: Set[Cell] = set()
        self.room_contact_walls: Set[Cell] = set()
        # Étape 1 : liste chaînée des points de parcours.
        self.path_start: Optional[PathNode] = None
        self.current: Optional[PathNode] = None
        self.active_issue: Optional[Issue] = None
        self.backtrack_target: Optional[PathNode] = None
        # Dernière position estimée de la zone de retour (capteur sémantique).
        self.return_area_sight: Optional[Point] = None
        self.modeexplore = True
        self.multiple_path = False
        self.state = "start"
        self.targets: List[Point] = []
        self.route: List[Point] = []
        self.survey_row = 0
        self.survey_origin_row = 0
        self.survey_direction = 1
        self.survey_row_step = 1
        self.survey_next: Optional[Tuple[int, int, int]] = None
        self.survey_shifting = False
        self.survey_failed: Set[Cell] = set()
        self.survey_tick = 0
        self.survey_stuck = 0
        self.warning: Optional[str] = None
        self.semantic_unavailable = False
        self.exact_position_warned = False
        self.forward_active = False
        self.progress_origin: Optional[Point] = None
        self.progress_heading: Optional[float] = None
        self.progress_ticks = 0
        self.progress_pause = 0

    def define_message_for_all(self):
        pass

    def _warn(self, message: str) -> None:
        if self.warning != message:
            logger.warning("Drone %s: %s", self.identifier, message)
            self.warning = message

    @staticmethod
    def _valid(values) -> bool:
        return values is not None and all(math.isfinite(value) for value in values)

    def _update_pose(self) -> bool:
        gps = self.measured_gps_position()
        compass = self.measured_compass_angle()
        odometer = self.odometer_values()
        valid_gps = self._valid(gps)
        valid_compass = compass is not None and math.isfinite(compass)
        valid_odometer = self._valid(odometer)
        old_heading = self.heading

        if valid_gps and gps is not None:
            if true_state_is_allowed():
                gps = self.true_position()
                if not self.exact_position_warned:
                    logger.warning("Drone %s: position exacte utilisée pour le débogage ; "
                                   "ce mode n'est pas autorisé en compétition",
                                   self.identifier)
                    self.exact_position_warned = True
                if not self._valid(gps):
                    self._warn("position exacte indisponible")
                    return False
            else:
                self.exact_position_warned = False
            self.estimated_position = (float(gps[0]), float(gps[1]))
        elif self.estimated_position is not None and old_heading is not None and valid_odometer:
            distance, relative_angle = odometer[:2]
            angle = old_heading + relative_angle
            self.estimated_position = (
                self.estimated_position[0] + distance * math.cos(angle),
                self.estimated_position[1] + distance * math.sin(angle),
            )
        else:
            self.estimated_position = None

        if valid_compass and compass is not None:
            self.heading = float(compass)
        elif old_heading is not None and valid_odometer:
            self.heading = normalize_angle(old_heading + odometer[2])
        else:
            self.heading = None
        return self.estimated_position is not None and self.heading is not None

    def _cell(self, point: Point) -> Cell:
        return math.floor(point[0] / self.CELL), math.floor(point[1] / self.CELL)

    def _cell_centre(self, cell: Cell) -> Point:
        return ((cell[0] + 0.5) * self.CELL, (cell[1] + 0.5) * self.CELL)

    def _scan(self, distances, angles, position: Point, heading: float, detections) -> None:
        free: Set[Cell] = {self._cell(position)}
        occupied: Set[Cell] = set()
        non_walls: Set[Cell] = set()
        entities = []
        if detections is not None:
            for data in detections:
                if (data.entity_type in (
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON,
                        DroneSemanticSensor.TypeEntity.DRONE,
                        DroneSemanticSensor.TypeEntity.RESCUE_CENTER,
                    ) and math.isfinite(data.angle) and math.isfinite(data.distance)
                        and data.distance > 0):
                    entities.append((data.angle, data.distance))
                elif (data.entity_type == DroneSemanticSensor.TypeEntity.OTHER
                        and math.isfinite(data.angle) and math.isfinite(data.distance)
                        and data.distance > 0):
                    # La zone de retour ("return area") n'est pas classée dans
                    # un type dédié par le capteur sémantique ; elle apparaît
                    # comme ``OTHER``. On garde sa dernière position estimée
                    # pour y entrer précisément à la fin de l'exploration.
                    angle = heading + data.angle
                    self.return_area_sight = (position[0] + data.distance * math.cos(angle),
                                              position[1] + data.distance * math.sin(angle))
        for distance, relative_angle in zip(distances, angles):
            if not math.isfinite(distance) or not math.isfinite(relative_angle) or distance <= 0:
                continue
            distance = min(float(distance), MAX_RANGE_LIDAR_SENSOR)
            angle = heading + relative_angle
            cos_angle, sin_angle = math.cos(angle), math.sin(angle)
            steps = max(1, math.ceil(distance / (self.CELL / 2)))
            for step in range(steps):
                length = distance * step / steps
                free.add(self._cell((position[0] + length * cos_angle,
                                     position[1] + length * sin_angle)))
            # Une mesure bruitée proche de la portée maximale n'est pas un mur fiable.
            if distance < MAX_RANGE_LIDAR_SENSOR - 10:
                cell = self._cell((position[0] + distance * cos_angle,
                                   position[1] + distance * sin_angle))
                occupied.add(cell)
                # Le capteur sémantique a moins de rayons que le lidar.
                if any(abs(normalize_angle(relative_angle - entity_angle))
                       <= self.SEMANTIC_ANGLE_TOLERANCE
                       and abs(distance - entity_distance) <= self.SEMANTIC_DISTANCE_TOLERANCE
                       for entity_angle, entity_distance in entities):
                    non_walls.add(cell)
        for cell in free - occupied:
            self.map[cell] = 1
            if self.room_map.get(cell) == 255:
                count = self.wall_clear_scans.get(cell, 0) + 1
                if count >= self.WALL_CLEAR_SCANS:
                    self.room_map[cell] = 1
                    self.wall_clear_scans.pop(cell, None)
                else:
                    self.wall_clear_scans[cell] = count
            else:
                self.room_map[cell] = 1
        for cell in occupied:
            self.map[cell] = 255
            self.wall_clear_scans.pop(cell, None)
            if detections is not None and cell not in non_walls:
                self.room_map[cell] = 255
                # Un mur confirmé plusieurs fois sans correspondance d'entité est
                # protégé des futures erreurs de classification (ex : bruit du
                # capteur sémantique près d'une entité comme le centre de secours).
                count = self.wall_confirm_scans.get(cell, 0) + 1
                if count >= self.WALL_CONFIRM_SCANS:
                    self.room_contact_walls.add(cell)
                    self.wall_confirm_scans.pop(cell, None)
                else:
                    self.wall_confirm_scans[cell] = count
            elif detections is None and self.room_map.get(cell) == 1:
                del self.room_map[cell]
        removed_wall = False
        for cell in non_walls:
            if cell in self.room_contact_walls:
                continue
            self.wall_clear_scans.pop(cell, None)
            self.wall_confirm_scans.pop(cell, None)
            if self.room_map.pop(cell, None) == 255:
                removed_wall = True
        for cell in self.contact_walls:
            self.map[cell] = 255
        for cell in self.room_contact_walls:
            self.room_map[cell] = 255
        if removed_wall and self.state == "survey" and self.current is not None:
            self.current.limits = (None, None, None, None)
            self.route = []

    def _wall_lines(self, vertical: bool) -> List[float]:
        occupied = [cell for cell, value in self.room_map.items() if value == 255]
        if not occupied:
            return []
        groups: Dict[int, Set[int]] = {}
        for x, y in occupied:
            axis, along = (x, y) if vertical else (y, x)
            for offset in (-1, 0, 1):
                groups.setdefault(axis + offset, set()).add(along)

        def longest_run(points: Set[int]) -> int:
            longest = run = 0
            previous = None
            for point in sorted(points):
                run = run + 1 if previous is not None and point == previous + 1 else 1
                longest = max(longest, run)
                previous = point
            return longest

        candidates = sorted(axis for axis, points in groups.items()
                            if longest_run(points) >= self.MIN_WALL_CELLS)
        clusters: List[List[int]] = []
        for axis in candidates:
            if not clusters or axis > clusters[-1][-1] + 1:
                clusters.append([axis])
            else:
                clusters[-1].append(axis)
        counts: Dict[int, int] = {}
        for x, y in occupied:
            axis = x if vertical else y
            counts[axis] = counts.get(axis, 0) + 1
        return [self.CELL * (max(cluster, key=lambda axis: counts.get(axis, 0)) + 0.5)
                for cluster in clusters]

    def _wall_limits(self, entry: Point) -> Tuple[Optional[float], Optional[float],
                                                 Optional[float], Optional[float]]:
        xs = self._wall_lines(vertical=True)
        ys = self._wall_lines(vertical=False)
        margin = self.CELL
        position = self.estimated_position if self.estimated_position is not None else entry
        left = [x for x in xs if x < min(entry[0], position[0]) - margin]
        right = [x for x in xs if x > max(entry[0], position[0]) + margin]
        bottom = [y for y in ys if y < min(entry[1], position[1]) - margin]
        top = [y for y in ys if y > max(entry[1], position[1]) + margin]
        detected = [max(left) if left else None, max(bottom) if bottom else None,
                    min(right) if right else None, min(top) if top else None]
        if self.current is None or self.current.entry != entry:
            return detected[0], detected[1], detected[2], detected[3]
        if self.current.incoming is not None:
            issue = self.current.incoming
            for index, direction in enumerate((issue.outward[0], issue.outward[1],
                                               -issue.outward[0], -issue.outward[1])):
                if direction > 0:
                    boundary = issue.centre[index % 2]
                    current = detected[index]
                    if current is None:
                        detected[index] = boundary
                    elif index < 2:
                        detected[index] = max(current, boundary)
                    else:
                        detected[index] = min(current, boundary)
        self.current.limits = detected[0], detected[1], detected[2], detected[3]
        return self.current.limits

    def _bounds(self, entry: Point) -> Optional[Bounds]:
        left, bottom, right, top = self._wall_limits(entry)
        if left is None or bottom is None or right is None or top is None:
            return None
        return left, bottom, right, top

    def _perimeter(self, bounds: Bounds) -> Tuple[List[Issue], List[Point]]:
        left, bottom, right, top = bounds
        issues: List[Issue] = []
        unobserved: List[Point] = []
        sides = (
            (True, left, bottom, top, (-1.0, 0.0)),
            (True, right, bottom, top, (1.0, 0.0)),
            (False, bottom, left, right, (0.0, -1.0)),
            (False, top, left, right, (0.0, 1.0)),
        )
        for vertical, fixed, lower, upper, normal in sides:
            fixed_cell = math.floor(fixed / self.CELL)
            first = math.floor(lower / self.CELL) + 2
            last = math.floor(upper / self.CELL) - 1
            opening: List[int] = []

            def finish_opening() -> None:
                if len(opening) >= self.MIN_EXIT_CELLS:
                    along = (opening[0] + opening[-1] + 1) * self.CELL / 2
                    centre = (fixed, along) if vertical else (along, fixed)
                    issues.append(Issue(centre, normal))
                opening.clear()

            for along in range(first, last):
                def at(offset: int) -> Cell:
                    return (fixed_cell + offset, along) if vertical else (along, fixed_cell + offset)

                wall = any(self.room_map.get(at(offset)) == 255 for offset in (-1, 0, 1))
                free = (self.room_map.get(at(-2)) == 1
                        and self.room_map.get(at(0)) == 1
                        and self.room_map.get(at(2)) == 1)
                if not wall and free:
                    opening.append(along)
                else:
                    finish_opening()
                    if not wall:
                        coordinate = (along + 0.5) * self.CELL
                        unobserved.append((fixed, coordinate) if vertical else (coordinate, fixed))
            finish_opening()
        return issues, unobserved

    def _route(self, start: Point, target: Point) -> Optional[List[Point]]:
        source, destination = self._cell(start), self._cell(target)
        if not self._clear_cell(destination):
            return None
        limits = None
        if self.state == "survey" and self.current is not None:
            limits = self._wall_limits(self.current.entry)
        queue = [(0, 0, source)]
        costs = {source: 0}
        previous: Dict[Cell, Cell] = {}
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost != costs[cell]:
                continue
            if cell == destination:
                cells = []
                while cell != source:
                    cells.append(cell)
                    cell = previous[cell]
                cells.reverse()
                path = []
                last = source
                direction = None
                for next_cell in cells:
                    step = next_cell[0] - last[0], next_cell[1] - last[1]
                    if direction is not None and step != direction:
                        path.append(self._cell_centre(last))
                    direction = step
                    last = next_cell
                if cells:
                    path.append(self._cell_centre(cells[-1]))
                return path + [target]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = cell[0] + dx, cell[1] + dy
                next_cost = cost + 1
                if not self._clear_cell(neighbour) or next_cost >= costs.get(neighbour, math.inf):
                    continue
                if limits is not None:
                    x, y = self._cell_centre(neighbour)
                    left, bottom, right, top = limits
                    if ((left is not None and x <= left + 20)
                            or (right is not None and x >= right - 20)
                            or (bottom is not None and y <= bottom + 20)
                            or (top is not None and y >= top - 20)):
                        continue
                costs[neighbour] = next_cost
                previous[neighbour] = cell
                heuristic = abs(neighbour[0] - destination[0]) + abs(neighbour[1] - destination[1])
                heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbour))
        return None

    def _clear_cell(self, cell: Cell) -> bool:
        if self.map.get(cell) != 1:
            return False
        x, y = cell
        return all(self.map.get((x + dx, y + dy)) != 255
                   for dx in range(-2, 3) for dy in range(-2, 3)
                   if dx * dx + dy * dy <= 4)

    def _set_targets(self, targets: List[Point]) -> None:
        self.targets = targets
        self.route = []

    def _start_survey(self, position: Point) -> None:
        self.survey_row = self._cell(position)[1]
        self.survey_origin_row = self.survey_row
        self.survey_direction = 1
        self.survey_row_step = 1
        self.survey_next = None
        self.survey_shifting = False
        self.survey_failed.clear()
        self.survey_stuck = 0
        self.survey_tick = self.SURVEY_INTERVAL

    def _survey_target(self, position: Point) -> Optional[Point]:
        assert self.current is not None
        left, bottom, right, top = self._wall_limits(self.current.entry)

        def eligible(cell: Cell) -> bool:
            x, y = self._cell_centre(cell)
            return (self.room_map.get(cell) == 1 and cell not in self.survey_failed
                    and self._clear_cell(cell)
                    and (left is None or x > left + 20)
                    and (right is None or x < right - 20)
                    and (bottom is None or y > bottom + 20)
                    and (top is None or y < top - 20))

        def reachable(cells: List[Cell]) -> Optional[Point]:
            for cell in cells:
                target = self._cell_centre(cell)
                if self._route(position, target) is not None:
                    return target
            return None

        if not self.survey_shifting:
            row = [cell for cell in self.room_map
                   if cell[1] == self.survey_row and eligible(cell)
                   and self.survey_direction * (self._cell_centre(cell)[0] - position[0])
                   > self.WAYPOINT_TOLERANCE * 2]
            row.sort(key=lambda cell: self.survey_direction * cell[0], reverse=True)
            target = reachable(row)
            if target is not None:
                return target
            self.survey_shifting = True

        for step in (self.survey_row_step, -1) if self.survey_row_step == 1 else (-1,):
            next_row = (self.survey_row + step * self.SURVEY_ROW_SPACING
                        if step == self.survey_row_step
                        else self.survey_origin_row - self.SURVEY_ROW_SPACING)
            cells = [cell for cell in self.room_map
                     if cell[1] == next_row and eligible(cell)]
            cells.sort(key=lambda cell: abs(self._cell_centre(cell)[0] - position[0]))
            target = reachable(cells)
            if target is not None:
                self.survey_next = (next_row, step, -self.survey_direction)
                return target
        return None

    def _track_progress(self, position: Point, heading: float) -> bool:
        moving = self.forward_active
        self.forward_active = False
        if self.progress_origin is None:
            if moving:
                self.progress_origin = position
                self.progress_heading = heading
            return False
        if math.dist(position, self.progress_origin) >= self.STALL_DISTANCE:
            self.progress_origin = position if moving else None
            self.progress_heading = heading if moving else None
            self.progress_ticks = 0
            self.progress_pause = 0
            return False
        if not moving:
            self.progress_pause += 1
            if self.progress_pause >= 10:
                self.progress_origin = None
                self.progress_heading = None
                self.progress_ticks = 0
                self.progress_pause = 0
            return False
        self.progress_pause = 0
        if (self.progress_heading is not None
                and abs(normalize_angle(heading - self.progress_heading)) > 0.5):
            self.progress_origin = position
            self.progress_heading = heading
            self.progress_ticks = 0
            return False
        self.progress_ticks += 1
        if self.progress_ticks < self.STALL_STEPS:
            return False

        # Avance sans changement de position : on considère l'endroit comme un mur.
        distance = self.OBSTACLE_DISTANCE + self.CELL / 2
        wall = self._cell((position[0] + distance * math.cos(heading),
                           position[1] + distance * math.sin(heading)))
        self.contact_walls.add(wall)
        self.room_contact_walls.add(wall)
        self.map[wall] = 255
        self.room_map[wall] = 255
        self.progress_origin = None
        self.progress_heading = None
        self.progress_ticks = 0
        self.route = []
        if self.state == "survey":
            if self.targets:
                self.survey_failed.add(self._cell(self.targets[0]))
            self.targets = []
            self.survey_next = None
        self._warn("aucune progression ; obstacle ajouté à la carte")
        return True

    @staticmethod
    def _stop() -> CommandsDict:
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    def _navigate(self, position: Point, heading: float, distances, angles) -> CommandsDict:
        while self.targets and math.dist(position, self.targets[0]) < self.WAYPOINT_TOLERANCE:
            if self.state == "survey" and self.survey_next is not None:
                self.survey_row, self.survey_row_step, self.survey_direction = self.survey_next
                self.survey_next = None
                self.survey_shifting = False
            self.targets.pop(0)
            self.route = []
        if not self.targets:
            return self._stop()
        if not self.route:
            route = self._route(position, self.targets[0])
            if route is None:
                self._warn("trajet indisponible dans la carte observée")
                if self.state == "survey":
                    self.survey_failed.add(self._cell(self.targets[0]))
                    self.targets = []
                    self.survey_next = None
                return self._stop()
            self.route = route
        while self.route and math.dist(position, self.route[0]) < self.WAYPOINT_TOLERANCE:
            self.route.pop(0)
        if not self.route:
            return self._stop()
        target = self.route[0]
        direction = math.atan2(target[1] - position[1], target[0] - position[0])
        difference = normalize_angle(direction - heading)
        turning = abs(difference) > 0.2
        blocked = not turning and any(
            math.isfinite(distance) and 0 < distance < self.OBSTACLE_DISTANCE
            and abs(normalize_angle(angle)) < 0.25
            for distance, angle in zip(distances, angles)
        )
        if blocked:
            self.route = []
            self._warn("obstacle devant le drone ; recalcul du trajet")

        ahead = [
            distance for distance, angle in zip(distances, angles)
            if math.isfinite(distance) and 0 < distance < self.SLOW_DISTANCE
            and abs(normalize_angle(angle)) < 0.35
        ]
        # Pleine vitesse si rien devant, puis ralentissement proportionnel à l'obstacle.
        cruise = self.MAX_SPEED
        if ahead:
            cruise = max(0.5, self.MAX_SPEED * (min(ahead) - self.OBSTACLE_DISTANCE)
                         / (self.SLOW_DISTANCE - self.OBSTACLE_DISTANCE))
        speed = 0.0 if turning or blocked else min(cruise, 0.15 * math.dist(position, target))

        velocity = self.measured_velocity()
        if velocity is not None and self._valid(velocity):
            forward_speed = velocity[0] * math.cos(heading) + velocity[1] * math.sin(heading)
            lateral_speed = -velocity[0] * math.sin(heading) + velocity[1] * math.cos(heading)
            forward = max(-1.0, min(1.0, 0.3 * (speed - forward_speed)))
            lateral = max(-1.0, min(1.0, -0.3 * lateral_speed))
            if not blocked:
                self.warning = None
        else:
            forward = 0.3 if speed > 0 else 0.0
            lateral = 0.0
            if not blocked:
                self._warn("odomètre indisponible ; navigation à vitesse réduite")
        self.forward_active = not turning and not blocked and forward > 0.05
        return {"forward": forward, "lateral": lateral,
                "rotation": (1.0 if difference > 0 else -1.0) if turning else 0.0,
                "grasper": 0}

    def _fallback_bounds(self) -> Optional[Bounds]:
        """Complète les côtés jamais détectés du rectangle par l'étendue
        déjà explorée (cellules libres connues), pour sortir d'un blocage."""
        assert self.current is not None
        left, bottom, right, top = self.current.limits
        free_cells = [cell for cell, value in self.room_map.items() if value == 1]
        if not free_cells:
            return None
        xs = [self._cell_centre(cell)[0] for cell in free_cells]
        ys = [self._cell_centre(cell)[1] for cell in free_cells]
        if left is None:
            left = min(xs) - self.CELL
        if right is None:
            right = max(xs) + self.CELL
        if bottom is None:
            bottom = min(ys) - self.CELL
        if top is None:
            top = max(ys) + self.CELL
        return left, bottom, right, top

    def _finalize_room(self, bounds: Bounds, issues: List[Issue]) -> None:
        """Étape 3 : enregistre le centre et les issues du nœud courant,
        à partir du périmètre connu (complet, ou forcé après blocage)."""
        assert self.current is not None
        self.current.bounds = bounds
        self.current.centre = ((bounds[0] + bounds[2]) / 2,
                               (bounds[1] + bounds[3]) / 2)
        if self.current.incoming is not None:
            issues = [issue for issue in issues
                     if math.dist(issue.centre, self.current.incoming.centre)
                     > self.CROSSING_DISTANCE]
        self.current.issues = issues
        if len(issues) > 1:
            self.multiple_path = True
        self.state = "centre"
        self.survey_next = None
        self._set_targets([self.current.centre])

    def _build_backtrack(self) -> List[Point]:
        """Étape 6 : remonte toutes les coordonnées de la liste chaînée."""
        assert self.current is not None
        chain: List[PathNode] = []
        node: Optional[PathNode] = self.current
        while node is not None:
            chain.append(node)
            node = node.previous
        points: List[Point] = []
        for index, node in enumerate(chain):
            if index > 0:
                points.append(node.centre if node.centre is not None else node.entry)
            points.append(node.entry)
        return points

    def _find_unvisited_ancestor(self) -> Optional[PathNode]:
        """Cherche, en remontant la liste chaînée, la dernière pièce qui a
        encore une issue non visitée (étape 6, cas ``multiple_path``)."""
        node = self.current.previous if self.current is not None else None
        while node is not None:
            if any(not issue.visited for issue in node.issues):
                return node
            node = node.previous
        return None

    def _build_backtrack_to(self, target: "PathNode") -> List[Point]:
        """Remonte la liste chaînée jusqu'au nœud ``target`` (inclus), afin de
        reprendre l'exploration à partir d'une de ses issues non visitées."""
        assert self.current is not None
        chain: List[PathNode] = []
        node: Optional[PathNode] = self.current
        while node is not None and node is not target:
            chain.append(node)
            node = node.previous
        points: List[Point] = []
        for index, node in enumerate(chain):
            if index > 0:
                points.append(node.centre if node.centre is not None else node.entry)
            points.append(node.entry)
        points.append(target.centre if target.centre is not None else target.entry)
        return points

    def control(self) -> CommandsDict:
        if self.state == "done":
            return self._stop()
        pose_available = self._update_pose()
        distances = self.lidar_values()
        angles = self.lidar_rays_angles()
        if (not pose_available or distances is None or angles is None
                or len(distances) != len(angles)
                or not any(math.isfinite(distance) and distance > 0 for distance in distances)):
            self.forward_active = False
            self.progress_origin = None
            self.progress_heading = None
            self.progress_ticks = 0
            self.progress_pause = 0
            self._warn("position, orientation ou lidar indisponible")
            return self._stop()
        assert self.estimated_position is not None and self.heading is not None
        position, heading = self.estimated_position, self.heading
        detections = self.semantic_values()
        if detections is None:
            if not self.semantic_unavailable:
                logger.warning("Drone %s: capteur sémantique indisponible ; "
                               "les nouveaux obstacles ne sont pas enregistrés comme murs",
                               self.identifier)
                self.semantic_unavailable = True
        else:
            self.semantic_unavailable = False
        self._scan(distances, angles, position, heading, detections)
        if self._track_progress(position, heading):
            return self._stop()

        # Étape 1 : enregistrement du point de départ.
        if self.state == "start":
            self.path_start = PathNode(entry=position)
            self.current = self.path_start
            self.modeexplore = True
            self.state = "survey"
            self._start_survey(position)

        assert self.current is not None
        # Étape 2 : balayage en zigzag pour délimiter le rectangle de la pièce.
        if self.state == "survey":
            self.survey_tick += 1
            if self.survey_tick >= self.SURVEY_INTERVAL:
                self.survey_tick = 0
                old_limits = self.current.limits
                bounds = self._bounds(self.current.entry)
                if self.current.limits != old_limits:
                    self.route = []
                issues, unobserved = self._perimeter(bounds) if bounds else ([], [])
                if bounds is not None and not unobserved:
                    self.survey_stuck = 0
                    self._finalize_room(bounds, issues)
                elif not self.targets:
                    target = self._survey_target(position)
                    if target is None:
                        self.survey_failed.clear()
                        self.survey_stuck += 1
                        if self.survey_stuck >= self.SURVEY_STUCK_LIMIT:
                            # Le zigzag ne peut plus progresser (ex : mur non
                            # détecté, zone inaccessible) ; on finalise la
                            # pièce avec le meilleur périmètre disponible
                            # plutôt que de rester bloqué indéfiniment.
                            self.survey_stuck = 0
                            forced_bounds = bounds if bounds is not None else self._fallback_bounds()
                            if forced_bounds is not None:
                                forced_issues, _ = self._perimeter(forced_bounds)
                                self._warn("périmètre incomplet ; pièce finalisée "
                                          "avec les données disponibles")
                                self._finalize_room(forced_bounds, forced_issues)
                            else:
                                self._warn("périmètre incomplet ; aucun trajet en zigzag accessible")
                        else:
                            self._warn("périmètre incomplet ; aucun trajet en zigzag accessible")
                    else:
                        self.survey_stuck = 0
                        self._set_targets([target])

        if self.targets:
            return self._navigate(position, heading, distances, angles)

        if self.state == "centre":
            remaining = [issue for issue in self.current.issues if not issue.visited]
            if remaining:
                # Étape 4 : direction vers la première issue enregistrée.
                self.active_issue = remaining[0]
                self.state = "exit"
                self._set_targets([self.active_issue.centre])
            else:
                ancestor = self._find_unvisited_ancestor()
                if ancestor is not None:
                    # Étape 6 (chemins multiples) : remonte jusqu'à la dernière
                    # pièce ayant une issue non visitée pour y reprendre
                    # l'exploration à partir de cette issue.
                    self.backtrack_target = ancestor
                    self.state = "backtrack_branch"
                    self._set_targets(self._build_backtrack_to(ancestor))
                else:
                    # Étape 6 : plus aucune issue non visitée nulle part ->
                    # retour au point de départ en remontant la liste chaînée.
                    self.modeexplore = False
                    self.state = "backtrack"
                    self._set_targets(self._build_backtrack())
        elif self.state == "exit":
            assert self.active_issue is not None
            self.active_issue.visited = True
            crossing = (self.active_issue.centre[0] + self.active_issue.outward[0] * self.CROSSING_DISTANCE,
                        self.active_issue.centre[1] + self.active_issue.outward[1] * self.CROSSING_DISTANCE)
            self.state = "cross"
            self._set_targets([crossing])
        elif self.state == "cross":
            assert self.active_issue is not None
            # Étape 4 (suite) : la position courante devient le nouveau point de
            # départ, nouveau nœud chaîné au précédent, qui est marqué visité.
            # Remarque : si la pièce précédente a plusieurs issues, plusieurs
            # nœuds enfants sont créés l'un après l'autre ; ``next`` ne garde
            # alors que le dernier, mais ``previous`` reste correct pour
            # chacun d'eux, ce qui suffit pour tous les retours en arrière.
            new_node = PathNode(entry=position, previous=self.current,
                                incoming=self.active_issue)
            self.current.next = new_node
            self.current.visited = True
            self.current = new_node
            self.room_map = {}
            self.wall_clear_scans.clear()
            self.wall_confirm_scans.clear()
            self.room_contact_walls.clear()
            self.active_issue = None
            self.modeexplore = True
            self.state = "survey"
            self._start_survey(position)
        elif self.state == "backtrack_branch":
            # De retour sur la pièce ancêtre : on reprend la boucle des
            # étapes 2-3-4 à partir de ses issues non visitées.
            assert self.backtrack_target is not None
            self.current = self.backtrack_target
            self.backtrack_target = None
            self.state = "centre"
        elif self.state == "backtrack":
            # Étape 6 (fin) : on a remonté la liste chaînée jusqu'au point de
            # départ. On entre précisément dans la zone de retour repérée au
            # capteur sémantique, plutôt que de se fier uniquement au point
            # GPS enregistré.
            if self.is_inside_return_area:
                self.state = "done"
            elif self.return_area_sight is not None:
                self.state = "enter_return_area"
                self._set_targets([self.return_area_sight])
            else:
                self.state = "done"
        elif self.state == "enter_return_area":
            if self.is_inside_return_area:
                self.state = "done"
            elif self.return_area_sight is not None:
                self._set_targets([self.return_area_sight])
            else:
                self.state = "done"
        return self._stop()
