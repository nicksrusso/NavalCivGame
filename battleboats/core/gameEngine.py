from typing import Dict, List, Optional, Tuple
import numpy as np

from .player import Player
from .map.Map import Map
from .shipyard.ship import Ship
from .shipyard.ship_type import ShipType
from .actions import (
    Action,
    MoveAction,
    AttackAction,
    BuildShipAction,
    BuildPortAction,
    CapturePortAction,
    EndTurnAction,
)


class gameEngine:
    """Sole owner of game state and all rules.

    The Map handles spatial truth, Players hold per-side bookkeeping, Ships
    are passive records — but every mutation flows through this class so
    invariants stay coordinated.
    """

    def __init__(self, map_json_path: str) -> None:
        self.map_json_path = map_json_path
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
        pass

    def _do_attack(self, a: AttackAction) -> None:
        pass

    def _do_build_ship(self, a: BuildShipAction) -> None:
        pass

    def _do_build_port(self, a: BuildPortAction) -> None:
        pass

    def _do_capture_port(self, a: CapturePortAction) -> None:
        pass

    def _do_end_turn(self, a: EndTurnAction) -> None:
        pass

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
        # TODO: pull stats from the CSV-backed base table once that loader exists.
        sid = self._new_ship_id()
        stats = None  # placeholder until ship_data is wired in
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
        """
        pass

    # ------------------------------------------------------------------ RL hooks
    def visible_enemy_ships(self, player_id: int) -> List[Ship]:
        pass

    def legal_actions(self, player_id: int) -> List[Action]:
        pass

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
