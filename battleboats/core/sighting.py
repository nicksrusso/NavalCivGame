from dataclasses import dataclass
from typing import Tuple

from .shipyard.ship_type import ShipType


@dataclass(slots=True)
class Sighting:
    """A player's record of an enemy ship's last-known state.

    `fresh=True` means the ship is currently within visibility range of one
    of the owning player's ships. `fresh=False` means the player has lost the
    view; `position` and `turn_seen` are frozen at the last fresh observation.
    Sightings persist until overwritten by a newer observation or cleared by
    a witnessed kill (see gameEngine._destroy_ship).
    """

    ship_id: int
    type: ShipType
    position: Tuple[int, int]
    turn_seen: int
    fresh: bool
