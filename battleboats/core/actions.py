"""Tagged action records consumed by gameEngine.step().

The env layer translates gym action tensors into these; the engine never
touches gym types.
"""
from dataclasses import dataclass
from typing import Tuple, Union
from .shipyard.ship_type import ShipType


@dataclass(frozen=True, slots=True)
class MoveAction:
    ship_id: int
    destination: Tuple[int, int]


@dataclass(frozen=True, slots=True)
class AttackAction:
    attacker_id: int
    target_id: int


@dataclass(frozen=True, slots=True)
class BuildShipAction:
    port: Tuple[int, int]
    ship_type: ShipType


@dataclass(frozen=True, slots=True)
class BuildPortAction:
    builder_ship_id: int


@dataclass(frozen=True, slots=True)
class CapturePortAction:
    landing_ship_id: int


@dataclass(frozen=True, slots=True)
class EndTurnAction:
    pass


Action = Union[
    MoveAction,
    AttackAction,
    BuildShipAction,
    BuildPortAction,
    CapturePortAction,
    EndTurnAction,
]
