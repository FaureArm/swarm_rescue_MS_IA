import heapq
import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle
from typing import Dict, List, Optional, Set, Tuple

from swarm_rescue.simulation.utils.constants import (
    FOV_SEMANTIC_SENSOR, MAX_RANGE_LIDAR_SENSOR, RESOLUTION_SEMANTIC_SENSOR,
)

from swarm_rescue.solutions.localisation import MeasuredState
from swarm_rescue.solutions.mapper_aristide import Mapper_Aristide

Point = Tuple[float, float]
Cell = Tuple[int, int]
Bounds = Tuple[float, float, float, float]

class Issue:
    """Une issue (ouverture) détectée sur le périmètre d'une pièce."""
    centre: Point
    outward: Point
    visited: bool = False


class PathNode:
    """Élément de la liste chaînée des points de parcours du drone."""
    entry: Point
    previous: Optional["PathNode"] = None
    next: Optional["PathNode"] = None
    incoming: Optional[Issue] = None
    centre: Optional[Point] = None
    issues: List[Issue] = []
    bounds: Optional[Bounds] = None
    limits: Tuple[Optional[float], Optional[float],
                  Optional[float], Optional[float]] = (None, None, None, None)
    visited: bool = False

class Explorer_Aristide:
    
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
    SEMANTIC_ANGLE_TOLERANCE = np.radians(FOV_SEMANTIC_SENSOR) / (
        RESOLUTION_SEMANTIC_SENSOR - 1
    )
    SEMANTIC_DISTANCE_TOLERANCE = 20.0

    
    def __init__(self, mapper:Mapper_Aristide, estimator:MeasuredState):
        self.mapper = mapper
        self.previous_waypoint = None
        self.journey = []
        self.estimator = estimator

        # Étape 1 : liste chaînée des points de parcours.
        self.path_start: Optional[PathNode] = None
        self.current: Optional[PathNode] = None
        self.active_issue: Optional[Issue] = None
        self.backtrack_target: Optional[PathNode] = None
        # Dernière position estimée de la zone de retour (capteur sémantique).
        self.return_area_sight: Optional[Point] = None
        self.state = "start"
        self.multiple_path = False
        self.targets: List[Point] = []
        self.route: List[Point] = []

        self.modeexplore = True
        
        self.survey_row = 0
        self.survey_origin_row = 0
        self.survey_direction = 1
        self.survey_row_step = 1
        self.survey_next: Optional[Tuple[int, int, int]] = None
        self.survey_shifting = False
        self.survey_failed: Set[Cell] = set()
        self.survey_tick = 0
        self.survey_stuck = 0

        # TODO : ajouter un update regulier de ces deux variables par l'estimateur
        self.estimated_position = None
        self.heading = None
        
    def _cell(self, point: Point) -> Cell:
        return np.floor(point[0] / self.CELL), np.floor(point[1] / self.CELL)

    def _cell_centre(self, cell: Cell) -> Point:
        return ((cell[0] + 0.5) * self.CELL, (cell[1] + 0.5) * self.CELL)

    def scan(self, distances, angles, position: Point, heading: float) -> None:
        free: Set[Cell] = {self._cell(position)}
        occupied: Set[Cell] = set()
        non_walls: Set[Cell] = set()

        for distance, relative_angle in zip(distances, angles):
            if not np.isfinite(distance) or not np.isfinite(relative_angle) or distance <= 0:
                continue

            distance = min(float(distance), MAX_RANGE_LIDAR_SENSOR)
            angle = heading + relative_angle

            cos_angle, sin_angle = np.cos(angle), np.sin(angle)

            steps = int(max(1, np.ceil(distance / (self.CELL / 2))))

            for step in range(steps):
                length = distance * step / steps
                free.add(self._cell((position[0] + length * cos_angle,
                                     position[1] + length * sin_angle)))
                
            # Une mesure bruitée proche de la portée maximale n'est pas un mur fiable.
            if distance < MAX_RANGE_LIDAR_SENSOR - 10:
                cell = self._cell((position[0] + distance * cos_angle,
                                   position[1] + distance * sin_angle))
                occupied.add(cell)
            # TODO: ajouter un controle sur les détections sémantiques pour 
            # ne pas les classer en murs
            #     
        for cell in free - occupied:
            self.mapper.map[cell] = 1
            if self.mapper.room_map.get(cell) == 255:
                count = self.mapper.wall_clear_scans.get(cell, 0) + 1
                if count >= self.WALL_CLEAR_SCANS:
                    self.mapper.room_map[cell] = 1
                    self.mapper.wall_clear_scans.pop(cell, None)
                else:
                    self.mapper.wall_clear_scans[cell] = count
            else:
                self.mapper.room_map[cell] = 1

        for cell in occupied:
            self.mapper.map[cell] = 255
            self.mapper.wall_clear_scans.pop(cell, None)
            
            if self.mapper.room_map.get(cell) == 1:
                del self.room_map[cell]

        removed_wall = False

        for cell in non_walls:
            if cell in self.mapper.room_contact_walls:
                continue

            self.mapper.wall_clear_scans.pop(cell, None)
            self.mapper.wall_confirm_scans.pop(cell, None)
            if self.mapper.room_map.pop(cell, None) == 255:
                removed_wall = True

        for cell in self.mapper.contact_walls:
            self.mapper.map[cell] = 255
        for cell in self.mapper.room_contact_walls:
            self.mapper.room_map[cell] = 255

        if removed_wall and self.current is not None:
            self.current.limits = (None, None, None, None)
            self.route = []

    def _wall_lines(self, vertical: bool) -> List[float]:
        occupied = [cell for cell, value in self.mapper.room_map.items() if value == 255]
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

        position = self.estimator.position_

        left = [x for x in xs if x < min(entry[0], position[0]) - margin]
        right = [x for x in xs if x > max(entry[0], position[0]) + margin]
        bottom = [y for y in ys if y < min(entry[1], position[1]) - margin]
        top = [y for y in ys if y > max(entry[1], position[1]) + margin]

        detected = [max(left) if left else None, max(bottom) if bottom else None,
                    min(right) if right else None, min(top) if top else None]

        if self.current is None or (self.current.entry != entry).all():
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
            fixed_cell = np.floor(fixed / self.CELL)
            first = np.floor(lower / self.CELL) + 2
            last = np.floor(upper / self.CELL) - 1
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

                wall = any(self.mapper.room_map.get(at(offset)) == 255 for offset in (-1, 0, 1))
                free = (self.mapper.room_map.get(at(-2)) == 1
                        and self.mapper.room_map.get(at(0)) == 1
                        and self.mapper.room_map.get(at(2)) == 1)
                
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
                if not self._clear_cell(neighbour) or next_cost >= costs.get(neighbour, np.inf):
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
        if self.mapper.map.get(cell) != 1:
            return False
        x, y = cell

        return all(self.mapper.map.get((x + dx, y + dy)) != 255
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
            return (self.mapper.room_map.get(cell) == 1 and cell not in self.survey_failed
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
            row = [cell for cell in self.mapper.room_map
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
            cells = [cell for cell in self.mapper.room_map
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
        
        if np.linalg.norm(position, self.progress_origin) >= self.STALL_DISTANCE:
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
        wall = self._cell((position[0] + distance * np.cos(heading),
                           position[1] + distance * np.sin(heading)))
        
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
                     if np.linalg.norm(issue.centre, self.current.incoming.centre)
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
    
    def select_exploration_waypoint(self):

        # Étape 1 : enregistrement du point de départ.
        if self.state == "start":
            self.path_start = PathNode()
            self.path_start.entry = self.estimator.position_
            self.current = self.path_start
            self.modeexplore = True
            self.state = "survey"
            self._start_survey(self.estimator.position_)

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

                    target = self._survey_target(self.estimator.position_)

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
                                self._finalize_room(forced_bounds, forced_issues)

                    else:
                        self.survey_stuck = 0
                        self._set_targets([target])

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
            new_node = PathNode(entry=self.estimator.position_, previous=self.current,
                                incoming=self.active_issue)
            self.current.next = new_node
            self.current.visited = True
            self.current = new_node
            self.mapper.room_map = {}
            self.mapper.wall_clear_scans.clear()
            self.mapper.wall_confirm_scans.clear()
            self.mapper.room_contact_walls.clear()
            self.active_issue = None
            self.modeexplore = True
            self.state = "survey"
            self._start_survey(self.estimator.position_)

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
            pass

        elif self.state == "enter_return_area":
            if self.is_inside_return_area:
                self.state = "done"
            elif self.return_area_sight is not None:
                self._set_targets([self.return_area_sight])
            else:
                self.state = "done"

        return self.targets
    
