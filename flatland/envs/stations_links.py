"""Data model stations and links, see https://flatland-association.github.io/flatland-book//environment/environment/stations_links.html"""
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
    track_number: int


@dataclass(frozen=True)
class Station:
    name: str
    gates: Dict[str, Gate]
    stopping_points: List[StoppingPoint]
    edges: List[IntVector2D]


@dataclass(frozen=True)
class Fibre:
    from_pin: str
    to_pin: str
    edges: List[IntVector2D]


@dataclass(frozen=True)
class Link:
    from_station: str
    from_gate: str
    from_facing: str
    to_station: str
    to_gate: str
    to_facing: str
    fibres: List[Fibre]


@dataclass(frozen=True)
class StationsLinks:
    stations: Dict[str, Station]
    links: List[Link]


@dataclass(frozen=True)
class GateRef:
    """Reference to a city's gate - (city index, facing direction)."""
    city: int
    direction: int


@dataclass(frozen=True)
class GateConnection:
    """Identifies an inter-city connection between two specific gate tracks."""
    from_station: int
    from_gate: int
    from_track: int
    to_station: int
    to_gate: int
    to_track: int
