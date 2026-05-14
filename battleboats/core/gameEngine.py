from typing import Dict, List, Optional, Tuple
import numpy as np

from .player import Player
from .map.Map import Map
from .shipyard.ship import Ship
from .shipyard.ship_type import ShipType
from .shipyard.ship_data import BASE_STATS, attack_modifier
from .actions import (
    Action,
    MoveAction,
    AttackAction,
    BuildShipAction,
    BuildPortAction,
    CapturePortAction,
    EndTurnAction,
)


STARTING_CASH: int = 250

_CARDINAL_OFFSETS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


class gameEngine:
    """Sole owner of game state and all rules.

    The Map handles spatial truth, Players hold per-side bookkeeping, Ships
    are passive records — but every mutation flows through this class so
    invariants stay coordinated.
    """

    def __init__(self, map_json_path: str, kill_curve_k: float = 2.0) -> None:
        self.map_json_path = map_json_path
        self.kill_curve_k = kill_curve_k
        self.map: Map = Map()
        self.players: List[Player] = []
        self.ships: Dict[int, Ship] = {}
        self.current_player: int = 0
        self.turn: int = 0
        self.winner: Optional[int] = None
        self.rng: np.random.Generator = np.random.default_rng()
        self._next_ship_id: int = 0
        self._dispatch = {
            MoveAction: self._do_move,
            AttackAction: self._do_attack,
            BuildShipAction: self._do_build_ship,
            BuildPortAction: self._do_build_port,
            CapturePortAction: self._do_capture_port,
            EndTurnAction: self._do_end_turn,
        }

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.map = Map()
        self.map.load(self.map_json_path)
        home_ports = self._infer_home_ports()
        self.players = [Player(0, home_ports[0]), Player(1, home_ports[1])]
        for p in self.players:
            p.cash = STARTING_CASH
        for pos in self.map.port_positions:
            owner = int(self.map.port_owner[pos])
            self.players[owner].owned_port_positions.add(pos)
            self.players[owner].port_materials.setdefault(pos, 0)
        self.ships = {}
        self._next_ship_id = 0
        self.current_player = 0
        self.turn = 0
        self.winner = None

    def _infer_home_ports(self) -> List[Tuple[int, int]]:
        homes: List[Optional[Tuple[int, int]]] = [None, None]
        for pos in self.map.port_positions:
            owner = int(self.map.port_owner[pos])
            if owner in (0, 1) and homes[owner] is None:
                homes[owner] = pos
        return [h for h in homes if h is not None]

    # ------------------------------------------------------------------ step
    def step(self, action: Action) -> None:
        if self.winner is not None:
            return
        self._dispatch[type(action)](action)

    # ------------------------------------------------------------------ action handlers
    def _do_move(self, a: MoveAction) -> None:
        ship = self.ships.get(a.ship_id)
        if ship is None or ship.owner != self.current_player:
            return
        if ship.tiles_moved_this_turn >= ship.stats.speed:
            return
        if not self.map.in_bounds(a.destination):
            return
        if self.map.manhattan(ship.position, a.destination) != 1:
            return
        if not self.map.is_water(a.destination):
            return
        if self.map.is_occupied(a.destination):
            return
        self.map.relocate_ship(ship.id, ship.position, a.destination)
        ship.position = a.destination
        ship.tiles_moved_this_turn += 1

    def _do_attack(self, a: AttackAction) -> None:
        attacker = self.ships.get(a.attacker_id)
        if attacker is None or attacker.owner != self.current_player:
            return
        if attacker.has_attacked:
            return
        if attacker.stats.attack_range <= 0:
            return
        defender = self.ships.get(a.target_id)
        if defender is None or defender.owner == self.current_player:
            return
        distance = self.map.manhattan(attacker.position, defender.position)
        if distance > attacker.stats.attack_range:
            return
        if distance > self._detection_distance(attacker, defender):
            return
        attacker.has_attacked = True
        if self._resolve_attack(attacker, defender):
            self._destroy_ship(defender.id)

    def _do_build_ship(self, a: BuildShipAction) -> None:
        # Silent no-op on illegal actions; legal_actions() will mask these out
        # for trained agents. Engine stays robust to bad input during dev / random play.
        player = self.players[self.current_player]
        if a.port not in player.owned_port_positions:
            return
        if not self.map.in_bounds(a.spawn_position):
            return
        if self.map.manhattan(a.spawn_position, a.port) != 1:
            return
        if not self.map.is_water(a.spawn_position):
            return
        if self.map.is_occupied(a.spawn_position):
            return
        cost = BASE_STATS[a.ship_type].cost
        if player.cash < cost:
            return
        player.cash -= cost
        self._spawn_ship(self.current_player, a.ship_type, a.spawn_position)

    def _do_build_port(self, a: BuildPortAction) -> None:
        pass

    def _do_capture_port(self, a: CapturePortAction) -> None:
        ship = self.ships.get(a.landing_ship_id)
        if ship is None or ship.owner != self.current_player:
            return
        if ship.type != ShipType.LANDING:
            return
        if not self.map.in_bounds(a.target):
            return
        if not self.map.is_port(a.target):
            return
        prev_owner_id = int(self.map.port_owner[a.target])
        if prev_owner_id == self.current_player:
            return
        if self.map.manhattan(ship.position, a.target) != 1:
            return
        self.map.set_port_owner(a.target, self.current_player)
        prev_owner = self.players[prev_owner_id]
        prev_owner.owned_port_positions.discard(a.target)
        prev_owner.port_materials.pop(a.target, None)
        captor = self.players[self.current_player]
        captor.owned_port_positions.add(a.target)
        captor.port_materials[a.target] = 0
        if a.target == prev_owner.home_port:
            self.winner = self.current_player

    def _do_end_turn(self, a: EndTurnAction) -> None:
        for sid in self.players[self.current_player].owned_ship_ids:
            self.ships[sid].reset_turn_flags()
        # self.turn counts rounds; increments after player 1 finishes their turn.
        if self.current_player == 1:
            self.turn += 1
        self.current_player = 1 - self.current_player

    # ------------------------------------------------------------------ internal helpers
    def _new_ship_id(self) -> int:
        sid = self._next_ship_id
        self._next_ship_id += 1
        return sid

    def _spawn_ship(
        self,
        owner: int,
        ship_type: ShipType,
        position: Tuple[int, int],
    ) -> Ship:
        sid = self._new_ship_id()
        stats = BASE_STATS[ship_type]
        ship = Ship(id=sid, type=ship_type, stats=stats, owner=owner, position=position)
        self.ships[sid] = ship
        self.players[owner].owned_ship_ids.add(sid)
        self.map.place_ship(sid, position)
        return ship

    def _destroy_ship(self, ship_id: int) -> None:
        ship = self.ships.pop(ship_id)
        self.players[ship.owner].owned_ship_ids.discard(ship_id)
        self.map.remove_ship(ship.position)

    def _detection_distance(self, scout: Ship, target: Ship) -> float:
        """Range at which `scout` can spot `target`. Asymmetric by design."""
        return scout.stats.scouting * target.stats.visibility

    def _resolve_attack(self, attacker: Ship, defender: Ship) -> bool:
        """Return True if defender is destroyed. Probability is a function of
        relative strength, modified by the attacker-vs-defender type matchup.
        Engine owns the RNG so rollouts are deterministic under a fixed seed.

        P(kill) = x^k / (1 + x^k),
            x = (attacker.strength * modifier) / defender.strength
            k = self.kill_curve_k (configurable; higher = more decisive).
        Defenders with strength 0 always die (and would otherwise divide by zero).
        """
        if defender.stats.strength <= 0:
            return True
        x = (attacker.stats.strength * attack_modifier(attacker.type, defender.type)) / defender.stats.strength
        xk = x ** self.kill_curve_k
        p = xk / (1.0 + xk)
        return float(self.rng.random()) < p

    # ------------------------------------------------------------------ RL hooks
    def visible_enemy_ships(self, player_id: int) -> List[Ship]:
        pass

    def enumerate_legal(self, player_id: int) -> List[Action]:
        """Concrete legal actions for `player_id` right now.

        Mirrors the validation in each `_do_*` handler so step() will accept
        every returned action. AttackAction is omitted until visibility lands
        in Phase 2. EndTurnAction is always included for the current player.
        """
        actions: List[Action] = []
        if self.winner is not None or player_id != self.current_player:
            return actions
        player = self.players[player_id]

        for sid in player.owned_ship_ids:
            ship = self.ships[sid]
            sx, sy = ship.position
            can_move = ship.tiles_moved_this_turn < ship.stats.speed
            is_builder = ship.type == ShipType.BUILDER
            is_landing = ship.type == ShipType.LANDING
            for dx, dy in _CARDINAL_OFFSETS:
                neighbor = (sx + dx, sy + dy)
                if not self.map.in_bounds(neighbor):
                    continue
                if can_move and self.map.is_water(neighbor) and not self.map.is_occupied(neighbor):
                    actions.append(MoveAction(sid, neighbor))
                if self.map.is_land(neighbor):
                    if is_builder and not self.map.is_port(neighbor):
                        actions.append(BuildPortAction(sid, neighbor))
                    if (
                        is_landing
                        and self.map.is_port(neighbor)
                        and int(self.map.port_owner[neighbor]) != player_id
                    ):
                        actions.append(CapturePortAction(sid, neighbor))

        affordable_types = [t for t, s in BASE_STATS.items() if player.cash >= s.cost]
        if affordable_types:
            for port in player.owned_port_positions:
                px, py = port
                for dx, dy in _CARDINAL_OFFSETS:
                    spawn = (px + dx, py + dy)
                    if not self.map.in_bounds(spawn):
                        continue
                    if not self.map.is_water(spawn):
                        continue
                    if self.map.is_occupied(spawn):
                        continue
                    for t in affordable_types:
                        actions.append(BuildShipAction(port, spawn, t))

        actions.append(EndTurnAction())
        return actions

    def legal_actions(self, player_id: int):
        """RL-friendly per-action-type mask. Deferred until env layer designs its action encoding."""
        raise NotImplementedError

    def get_state(self) -> dict:
        """Full ground-truth state. The env layer applies fog-of-war masking."""
        pass

    def is_terminal(self) -> bool:
        return self.winner is not None

    def clone(self) -> "gameEngine":
        """Fast snapshot for RL rollouts / search. Copies numpy grids and the
        ship registry without going through reset()/JSON load.
        """
        pass
