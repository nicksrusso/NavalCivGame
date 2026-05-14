from dataclasses import dataclass
from typing import Tuple
from .ship_type import ShipType
from .ship_stats import ShipStats


@dataclass(slots=True)
class Ship:
    """Per-instance ship record. Stats are frozen at build time from owner's tech."""

    id: int
    type: ShipType
    stats: ShipStats
    owner: int
    position: Tuple[int, int]
    tiles_moved_this_turn: int = 0
    has_attacked: bool = False

    def reset_turn_flags(self) -> None:
        self.tiles_moved_this_turn = 0
        self.has_attacked = False
