import numpy as np
from swarm_rescue.simulation.utils.utils import normalize_angle
from typing import Dict, List, Optional, Set, Tuple

from swarm_rescue.simulation.utils.constants import (
    FOV_SEMANTIC_SENSOR, MAX_RANGE_LIDAR_SENSOR, RESOLUTION_SEMANTIC_SENSOR,
)

from swarm_rescue.solutions.localisation import MeasuredState
from swarm_rescue.solutions.mapper import LidarMapper

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

class Explorer:
    
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

    
    def __init__(self, mapper:LidarMapper):
        self.mapper = mapper
        self.previous_waypoint = None
        self.journey = []

        
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

    def compute_next_step(self, estimator:MeasuredState):
        if self.state == "done":
            return self._stop()

        position = estimator.position_

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
