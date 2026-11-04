# Travel Wise Data Model

> Copied from flatland-book (`resources/travelwise/data_model.md`, 2724ac39) and updated to the decisions of the
> passenger env spike, see [the plan](../plans/travelwise-passenger-env.md) (PR #528).

This page gives technical details to the general approach of [Travel Wise](../travelwise.md).

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
            next_stop
            vehicles_to_board
            vehicle_stops_at_next_stop
        }
        class PassengerPolicy {
            act()
        }
    }

```
