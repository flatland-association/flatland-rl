"""Data model stations and links, see https://flatland-association.github.io/flatland-book/environment/environment/stations_links.html
Convention: use fully qualified names.
"""
from dataclasses import dataclass
from typing import Dict, List

from flatland.core.grid.grid_utils import IntVector2D


@dataclass(frozen=True)
class Pin:
    # A.N.0, A.N.1, ...
    name: str
    node: IntVector2D


@dataclass(frozen=True)
class Gate:
    # A.N, A.S, ...
    name: str
    pins: Dict[int, Pin]


@dataclass(frozen=True)
class StoppingPoint:
    # A.0, A.1, ...
    name: str
    node: IntVector2D


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
    edges: List[IntVector2D]


@dataclass(frozen=True)
class Link:
    # A.N.0, A.N.1, ...
    from_pin: str
    to_pin: str
    fibres: List[Fibre]


@dataclass(frozen=True)
class StationsLinks:
    stations: Dict[str, Station]
    links: List[Link]
