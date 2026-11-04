# Travel Wise Data Model

> Copied from flatland-book (`resources/travelwise/data_model.md`, 2724ac39) and updated to the decisions of the
> passenger env spike, see [the plan](../plans/travelwise-passenger-env.md) (PR #528).

This page gives technical details to the general approach of [Travel Wise](../travelwise.md).

## Glossary

The terms of [Travel Wise](../travelwise.md) (its introduction and its environments), with the class implementing each (as in the diagram below)
and its implementation status in flatland-rl: ✅ implemented, 💡 partly implemented, 🔴 open.

| Term | Description | Implementing class | Status |
|---|---|---|---|
| journey | A journey is a travel from point A to point B using potentially different modes of transportation. | `PassengerJourney` | 💡 origin, given transfers and destination with their times; no modes of transportation (any vehicle of the rail env) |
| itinerary | The itinerary describes the planned route (space and time) for a given journey. The initial itinerary might be updated during the scenario according to a given metric (default: shortest time). | - | 🔴 not modelled: the passenger policy (`PassengerPolicy`) picks the vehicle at each stop at runtime |
| transfer | Each change of mode of transportation is a transfer, even if the mode is the same, e.g. from one train to another. | `PassengerJourney.waypoints[1:-1]`, `LineFlexibleStoppingPoint` | 💡 the groups between origin and destination, each a `LineFlexibleStoppingPoint`; given, not planned; no minimum transfer time; no walkways |
| leg | The part of a journey on one vehicle, between two transfers (in flatland-rl: a ride). | `PassengerJourney.waypoints[i:i + 2]` | 💡 two consecutive groups; the vehicle ridden is not recorded, and a ride may use several vehicles (e.g. forced off at a vehicle's final stop) |
| route | In flatland-rl, a vehicle's route is every entry point it passes, in order (on the grid each cell entered, with its direction; on the graph each node), from its initial entry point through one alternative per stop. Its stops are a few of these entries. Not the passenger's itinerary. | `FlatlandPolicy`, `ShortestPathPolicy`, `Router` | ✅ the auto-pilot (`ShortestPathPolicy`) follows the route its `Router` finds |
| `InfrastructureEnv` | The infrastructure layer: a generalized version of the `RailEnv`, a graph representation of the infrastructure with different transportation modes as properties of the edges, including walkways. | `RailEnv`, `GraphRailEnv` | 💡 the rail env on the graph view; no modes of transportation on the edges, no walkways |
| `PassengerEnv` | The passenger layer: all properties correspond to those of the `InfrastructureEnv` (inheritance in OO terms), and it depends on an existing `InfrastructureEnv` to be generated. | `PassengerEnv` | ✅ composition instead of inheritance: it holds the rail env and implements `Environment` for the passengers |
| `TravelWiseEnv` | `InfrastructureEnv` and `PassengerEnv` together. | `PassengerEnv` | ✅ `PassengerEnv` steps the vehicles with the auto-pilot, so it plays this role; no class of its own |
| infrastructure (`InfrastructureEnv`) | Graph consisting of nodes and multimodal edges. | `TransitionMap`, `GraphTransitionMap` | 💡 nodes are entry points; edges carry no mode of transportation |
| passenger graph (`PassengerEnv`) | The passenger's own graph, a subgraph of the infrastructure graph. | - | 🔴 not modelled: the passenger sees the vehicles it can board and their planned stops |
| line (`InfrastructureEnv`) | List of stations that are served successively. | `Line`, `LineFlexibleStoppingPoint` | ✅ a vehicle's waypoint groups (`EnvAgent.waypoints`), each with alternative stops; stations are not modelled |
| timetable (`InfrastructureEnv`) | List of time windows (latest arrival, earliest departure) for a given line. | `Timetable`, `AgentTimetable` | ✅ |
| action (`InfrastructureEnv`) | Generalized version of the railway case actions, including _move to edge a1_, _accelerate_, _brake_ (a larger action space, since there can be more than two outgoing edges at a node). | `RailEnvActions` | 💡 the railway actions (left, forward, right, stop), also on the graph; no generalized _move to edge_ |
| action (`PassengerEnv`) | In principle the same actions; in practice only _move to edge a1_, since passengers leave their graph as soon as they _hop on_ a vehicle. | `PassengerAction` | ✅ `WAIT`, `HOP_ON` a given vehicle, `HOP_OFF`; hopping on is an explicit action, and passengers do not walk |
| observation (`InfrastructureEnv`) | Conventional `RailEnv` observations, e.g. tree observation. | `ShortestPathObservation`, `VehicleObservation` | ✅ the auto-pilot's own observation, with the positions and plans of all vehicles; the conventional observations (e.g. tree observation) remain for `RailEnv` on the grid |
| observation (`PassengerEnv`) | Global observation of the infrastructure graph and timetables; potentially a list of itineraries to choose from. | `PassengerObservation` | 💡 its own situation and the vehicles it can board now with their planned stops; the map, timetables and journeys come with the infos; no global view of all vehicles' future stops (query view), no itineraries |
| event (`InfrastructureEnv`, `PassengerEnv`) | Triggers for milestones, e.g. delayed trains, trains passing a specific node, or passenger specific: _my_ train is delayed. | - | 🔴 not modelled |
| reward (`InfrastructureEnv`) | Conventional `RailEnv` rewards. | `Rewards`, `DefaultRewards` | ✅ |
| reward (`PassengerEnv`) | - | `PassengerEnv` | ✅ arriving before the latest arrival is rewarded; arriving late or not at all and boarding before the earliest departure are penalized |
| malfunction (`InfrastructureEnv`) | Includes breakdowns, departure delays, ... | `MalfunctionGenerator` | ✅ also on the graph view; the auto-pilot does not react to them (it waits on its fixed route) |

## Building Blocks

Flatland represents RL view of the world: agent policies act on the env upon receiving a (partial) observation of the env state.
The world is run by a controller loop and the run is collected in a trajectory.

External effects can be modelled by effects generators, modifying the environment state before or after an env step, reflecting e.g.

- random disruptions of trains (due to door not closing, passenger delaying departure etc.)
- availability of infrastructure (e.g. opening a passenger fast-lane at airports)

The railway domain is reflected at the microscopic level:

- infrastructure/topology is defined by the transition map and stations and stops in the map
- services are defined spatially by lines and spatio-temporally by timetables

The Travel Wise extension `PassengerEnv` reflects passenger journeys.

Technically, it implements the `Environment` interface for the passengers and holds an underlying rail env
(`GraphRailEnv`) where trains, ships etc. run - composition, not inheritance.

The passenger env steps the rail env itself, with an "auto-pilot" Flatland policy for the vehicles
(`ShortestPathPolicy`): in each step it first applies the passengers' actions, then steps the rail env, and the riding
passengers move with their vehicles. So passengers and vehicles decide on the same state.
The auto-pilot drives each vehicle along a route: every entry point the vehicle passes, in order (on the grid each
cell entered, with its direction; on the graph each node), through one alternative of each waypoint group (aka.
`LineFlexibleStoppingPoint`) of its line. The waypoints are a few entries of the route.
The Travel Wise policy runner holds the passenger policy and steps the passenger env only, with the passengers'
actions based on their observations of the previous step.
Agents controlled in rail env are vehicles, agents controlled in the passenger env are passengers.

A passenger waits at an entry point or rides a vehicle. Its actions are `WAIT`, `HOP_ON` a given vehicle and
`HOP_OFF`; the env applies an action if it is feasible and ignores it otherwise. It observes the vehicles it can board
and their planned stops (`TimetablePredictor`); static data (map, timetables, journeys) comes with the infos.

```mermaid
classDiagram
    direction LR
    note for PassengerEnv "Agents reflect passengers, waiting at an entry point or riding a vehicle. Actions: WAIT, HOP_ON, HOP_OFF."
    note for RailEnv "Agents reflect trains/ships/etc. Configurations are entry points: graph nodes or grid cell entry points (r,c,d)."
    note for TransitionMap "Topology"
    note for Line "Services in Space; passenger journeys have the same shape"
    note for Timetable "Services in Space and Time"
    note for StoppingPoint "aka. Haltepunkt. "
    RailEnv --> TransitionMap
    AgentTimetable "1" --> "1" Line
    AgentTimetable "1" --> "2..." AgentTimetableItem: timetable
    PassengerEnv --> RailEnv : rail_env, passengers use trains, ships etc.
    PassengerEnv --> FlatlandPolicy : vehicle_policy, auto-pilot
    PassengerEnv "1" --> "1.." PassengerJourney : passenger_journeys
    PassengerJourney "1" --> "2.." LineFlexibleStoppingPoint : waypoints
    StoppingPoint "1.." --> "1" Station
    StoppingPoint "0,1" --> "1" Coordinate
    Line "1" --> "1.." LineFlexibleStoppingPoint
    LineFlexibleStoppingPoint "1.." --> "1.." StoppingPoint
    FlatlandPolicy --> RailEnv
    PassengerPolicy --> PassengerEnv
    TravelWisePolicyRunner --> PassengerPolicy
    TravelWisePolicyRunner --> PassengerEnv
    PassengerPolicy --> PassengerObservation : acts on
    PassengerPolicy --> PassengerAction : returns

    namespace Controller {
        class TravelWisePolicyRunner {
        }
    }

    namespace Flatland {
        class RailEnv {
            effects_generator
            events_generator
            step() "env step, called by controller"
            reset() "re-generate a new rail, line and timetable"
            register() "register for event topics"
            deregister() "de-register from event topics"
        }

        class TransitionMap {
            configurations: Set[Coordinate] "aka. nodes aka. grid cell entry points"
            transitions: Set[Tuple[Coordinate, Coordinate]] "aka. edges aka. grid cell transition"
            apply_action_independent()
            get_successor_coordinates()
            get_predecessor_coordinates()
            is_valid_coordinate()
        }

        class Station {
            description: Any
        }

        class StoppingPoint {
            description: Any
        }

        class Line

        class LineFlexibleStoppingPoint

        class Timetable {
            agentTimetables: Map[EnvAgent, AgentTimetable]
        }

        class AgentTimetable {
            line: Line
            initial(): Configuration
            targets(): Set[Configuration]
            timetable: Map[LineFlexibleStop, AgentTimetableItem]
        }

        class AgentTimetableItem {
            latesttArrival: int
            earliestDeparture: int
        }

        class FlatlandPolicy {
            <<interface>>
            act()
        }

        class EnvAgent {
            current(): Optional[Configuration]
            next_configuration(): Optional[Configuration]
            offset: Fraction
            timetable: AgentTimeTable
            next_stop(): LineFlexibleStop
        }
        class Coordinate
    }
    namespace TravelWise {
        class PassengerEnv {
            step()
            next_stop()
        }
        class PassengerJourney {
            waypoints
            waypoints_earliest_departure
            waypoints_latest_arrival
        }
        class PassengerAction {
            <<enumeration>>
            WAIT
            HOP_ON
            HOP_OFF
        }
        class PassengerObservation {
            mode
            stop_index
            vehicles_to_board
            vehicle_stops_at_next_stop
        }
        class PassengerPolicy {
            act()
        }
    }

```
