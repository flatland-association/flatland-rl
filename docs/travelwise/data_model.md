# Travel Wise Data Model

This page gives technical details to the general approach of [Travel Wise](../travelwise).

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

Technically, it is an extension of RailEnv, but also refers to an underlying RailEnv where trains, ships
etc. run.

Travel Wise policy runner steps through the rail env by using an "auto-pilot" Flatland policy for the rail env and then a passenger policy to step the passenger
env, based on the observed new state of the rail env.
Agents controlled in rail env are vehicles, agents controlled in the passenger env are passengers.

```mermaid
classDiagram
    direction LR
    note for PassengerEnv "Agents reflect passengers and configurations represent passenger locations and policies drive passengers; TODO: what are the actions"
    note for RailEnv "Agents reflect trains/ships/etc. Configurations can be graph nodes or grid cell entry points (r,c,d)."
    note for TransitionMap "Topology"
    note for Line "Services in Space; aka. Journey for passengers"
    note for Timetable "Services in Space and Time; aka. Initial Itinerary for passengers"
    note for StoppingPoint "aka. Haltepunkt. "
    RailEnv --> TransitionMap
    AgentTimetable "1" --> "1" Line
    AgentTimetable "1" --> "2..." AgentTimetableItem: timetable
    PassengerEnv --|> "inheritance" RailEnv
    PassengerEnv --> "passengers use trains, ships etc." RailEnv
    StoppingPoint "1.." --> "1" Station
    StoppingPoint "0,1" --> "1" Coordinate
    Line "1" --> "1.." LineFlexibleStoppingPoint
    LineFlexibleStoppingPoint "1.." --> "1.." StoppingPoint
    FlatlandPolicy --> RailEnv
    PassengerPolicy --> PassengerEnv
    TravelWisePolicyRunner --> FlatlandPolicy
    TravelWisePolicyRunner --> PassengerPolicy

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
        }
        class PassengerPolicy {
            act()
        }
    }

```
