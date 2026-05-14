from typing import Dict, Set, Tuple

from .sighting import Sighting


class Player:
    """Per-side bookkeeping: cash, owned units, port stockpiles, enemy memory.

    Pure data container. All world mutation goes through gameEngine — Player
    never touches the map or other players.
    """

    def __init__(self, id: int, home_port: Tuple[int, int]) -> None:
        self.id = id
        self.home_port = home_port
        self.cash: int = 0
        self.owned_ship_ids: Set[int] = set()
        self.owned_port_positions: Set[Tuple[int, int]] = {home_port}
        self.port_materials: Dict[Tuple[int, int], int] = {home_port: 0}
        self.sightings: Dict[int, Sighting] = {}
