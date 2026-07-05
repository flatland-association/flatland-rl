"""Data model stations and links, see https://flatland-association.github.io/flatland-book//environment/environment/stations_links.html
Conventions:
- use fully qualified names
- use `_idx` for relative names corresponding to dict keys.
"""
from dataclasses import dataclass
from typing import Dict, List

from flatland.core.grid.grid_utils import IntVector2D


@dataclass(frozen=True)
class Pin:
    node: IntVector2D
    name: str


@dataclass(frozen=True)
class Gate:
    name: str
    pins: Dict[int, Pin]


@dataclass(frozen=True)
class StoppingPoint:
    node: IntVector2D
    name: str
    # relative to station
    stopping_point_idx: int


@dataclass(frozen=True)
class Station:
    # A, B, ..., Z, AA, AB, .., ZZ, ..
    name: str
    # N, E, S, W
    gates: Dict[str, Gate]
    stopping_points: List[StoppingPoint]
    edges: List[IntVector2D]


@dataclass(frozen=True)
class Fibre:
    # A.N.0, A.N.1, ...
    from_pin: str
    to_pin: str
    edges: List[IntVector2D]


@dataclass(frozen=True)
class Link:
    # A, B, ..., Z, AA, AB, .., ZZ, ..
    from_station: str
    # A.N, A.S, ...
    from_gate: str
    # N, E, S, W
    from_gate_idx: str
    to_station: str
    to_gate: str
    to_gate_idx: str
    fibres: List[Fibre]


@dataclass(frozen=True)
class StationsLinks:
    stations: Dict[str, Station]
    links: List[Link]


@dataclass(frozen=True)
class GateRef:
    city: int
    direction: int


@dataclass(frozen=True)
class GateConnection:
    from_station: int
    from_gate: int
    from_track: int
    to_station: int
    to_gate: int
    to_track: int
