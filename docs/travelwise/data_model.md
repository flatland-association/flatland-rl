# Travel Wise Data Model

> Copied from flatland-book (`resources/travelwise/data_model.md`, 2724ac39) and updated to the decisions of the
> passenger env spike, see [the plan](../plans/travelwise-passenger-env.md) (PR #528).
>
> TODO port back to flatland-book (`resources/travelwise/data_model.md`), once PR #528 is merged:
>
> - the updates to the spike's decisions: composition over the rail env, passenger actions, journeys and
>   observations, the runner, a vehicle's route, the glossary;
> - the diagram shows what is implemented, with flatland-rl's names (entry points, `EnvAgent`'s fields, `Line`,
>   `Timetable`, `Policy`, `BoardForTargetPassengerPolicy`; no `AgentTimetable`), marks the target design, and adds `ResourceMap`, `MalfunctionGenerator`, `ObservationBuilder`, `PredictionBuilder`,
>   `Rewards` and `Trajectory`;
> - the glossary names the TravelWise concepts not built yet (passenger state, group size, modes of transportation,
>   conditional paths, capacities, milestones, the demand generator), and the diagram notes the roles of
>   `InfrastructureEnv` and `TravelWiseEnv` and what a journey is;
> - the passenger rewards class (`PassengerRewards`).

This page gives technical details to the general approach of [Travel Wise](../travelwise.md).

## Glossary

The terms of [Travel Wise](../travelwise.md) (its introduction and its environments), with the class implementing each (as in the diagram below)
and its implementation status in flatland-rl: ✅ implemented, 💡 partly implemented, 🔴 open.

| Term | Description | Implementing class | Status |
|---|---|---|---|
| journey | A journey is a travel from point A to point B using potentially different modes of transportation. | `PassengerJourney` | 💡 origin, given transfers and destination with their times; no modes of transportation (any vehicle of the rail env) |
| itinerary | The itinerary describes the planned route (space and time) for a given journey. The initial itinerary might be updated during the scenario according to a given metric (default: shortest time). | - | 🔴 not modelled: the passenger policy (`BoardForTargetPassengerPolicy`) picks the vehicle at each stop at runtime; planning it needs a query view of all vehicles' future stops ([#562](https://github.com/flatland-association/flatland-rl/issues/562)) |
| transfer | Each change of mode of transportation is a transfer, even if the mode is the same, e.g. from one train to another. | `PassengerJourney.waypoints[1:-1]`, `LineFlexibleStoppingPoint` | 💡 the groups between origin and destination, each a `LineFlexibleStoppingPoint`; given, not planned; no minimum transfer time; no walkways |
| leg | The part of a journey on one vehicle, between two transfers (in flatland-rl: a ride). | `PassengerJourney.waypoints[i:i + 2]` | 💡 two consecutive groups; the vehicle ridden is not recorded, and a ride may use several vehicles (e.g. forced off at a vehicle's final stop) |
| route | In flatland-rl, a vehicle's route is every entry point it passes, in order (on the grid each cell entered, with its direction; on the graph each node), from its initial entry point through one alternative per stop. Its stops are a few of these entries. Not the passenger's itinerary. | `Policy`, `ShortestPathPolicy`, `Router` | ✅ the auto-pilot (`ShortestPathPolicy`) follows the route its `Router` finds |
| `InfrastructureEnv` | The infrastructure layer: a generalized version of the `RailEnv`, a graph representation of the infrastructure with different transportation modes as properties of the edges, including walkways. | `RailEnv`, `GraphRailEnv` | 💡 the rail env on the graph view; no modes of transportation on the edges, no walkways |
| `PassengerEnv` | The passenger layer: all properties correspond to those of the `InfrastructureEnv` (inheritance in OO terms), and it depends on an existing `InfrastructureEnv` to be generated. | `PassengerEnv` | ✅ composition instead of inheritance: it holds the rail env and implements `Environment` for the passengers |
| `TravelWiseEnv` | `InfrastructureEnv` and `PassengerEnv` together. | `PassengerEnv` | ✅ `PassengerEnv` steps the vehicles with the auto-pilot, so it plays this role; no class of its own |
| infrastructure (`InfrastructureEnv`) | Graph consisting of nodes and multimodal edges. | `TransitionMap`, `GraphTransitionMap` | 💡 nodes are entry points; edges carry no mode of transportation |
| mode of transportation (`InfrastructureEnv`) | Train, bus, plane, ship, walkway, ..., as properties of the edges. | - | 🔴 not modelled: rail only, no walkways |
| conditional path (`InfrastructureEnv`) | Optional paths and the conditions under which they can be used, e.g. a shorter walkway for passengers with reduced mobility, or fast-track lanes for delayed passengers. | - | 🔴 not modelled |
| capacity (`InfrastructureEnv`) | Capacities of (some) modes of transportation. | - | 🔴 not modelled: any number of passengers rides a vehicle |
| passenger graph (`PassengerEnv`) | The passenger's own graph, a subgraph of the infrastructure graph. | - | 🔴 not modelled: the passenger sees the vehicles it can board and their planned stops |
| passenger (`PassengerEnv`) | A group of people (N=1..n). | `PassengerEnv` agent (`handle`) | 💡 one agent per passenger; no group size |
| passenger state (`PassengerEnv`) | Where a passenger is: at a stopping point while waiting, on a vehicle while riding; it changes by hopping on and off. | `PassengerMode` | 💡 off the map, waiting at a resource, or riding a vehicle; no passenger configuration (the entry point instead of the resource); one object per passenger (`Passenger`) is planned |
| line (`InfrastructureEnv`) | List of stations that are served successively. | `Line`, `LineFlexibleStoppingPoint` | ✅ a vehicle's waypoint groups (`EnvAgent.waypoints`), each with alternative stops; stations are not modelled |
| timetable (`InfrastructureEnv`) | List of time windows (latest arrival, earliest departure) for a given line. | `Timetable`, `EnvAgent.waypoints_earliest_departure`, `EnvAgent.waypoints_latest_arrival` | ✅ |
| action (`InfrastructureEnv`) | Generalized version of the railway case actions, including _move to edge a1_, _accelerate_, _brake_ (a larger action space, since there can be more than two outgoing edges at a node). | `RailEnvActions` | 💡 the railway actions (left, forward, right, stop), also on the graph; no generalized _move to edge_ |
| action (`PassengerEnv`) | In principle the same actions; in practice only _move to edge a1_, since passengers leave their graph as soon as they _hop on_ a vehicle. | `PassengerAction` | ✅ `WAIT`, `HOP_ON` a given vehicle, `HOP_OFF`; hopping on is an explicit action, and passengers do not walk |
| observation (`InfrastructureEnv`) | Conventional `RailEnv` observations, e.g. tree observation. | `ShortestPathObservation`, `VehicleObservation` | ✅ the auto-pilot's own observation, with the positions and plans of all vehicles; the conventional observations (e.g. tree observation) remain for `RailEnv` on the grid |
| observation (`PassengerEnv`) | Global observation of the infrastructure graph and timetables; potentially a list of itineraries to choose from. | `PassengerObservation` | 💡 its own situation and the vehicles it can board now with their planned stops; the map, timetables and journeys come with the infos; no global view of all vehicles' future stops (query view, [#562](https://github.com/flatland-association/flatland-rl/issues/562)), no itineraries |
| event (`InfrastructureEnv`, `PassengerEnv`) | Triggers for milestones, e.g. delayed trains, trains passing a specific node, or passenger specific: _my_ train is delayed. | - | 🔴 not modelled; waits for an event generator in Flatland ([#424](https://github.com/flatland-association/flatland-rl/issues/424)) |
| milestone (`PassengerEnv`) | Passed on a journey; passing it triggers communication with the Travel Wise layer, which is above and outside Flatland. | - | 🔴 not modelled; builds on events |
| reward (`InfrastructureEnv`) | Conventional `RailEnv` rewards. | `Rewards`, `DefaultRewards` | ✅ |
| reward (`PassengerEnv`) | What the passenger policy optimizes, e.g. shortest time, fewest transfers, longer transfer times (e.g. for passengers with reduced mobility). | `PassengerRewards`, `DefaultPassengerRewards` | 💡 arriving before the latest arrival is rewarded; arriving late or not at all and boarding before the earliest departure are penalized; transfers and transfer times are not rewarded |
| malfunction (`InfrastructureEnv`) | Includes breakdowns, departure delays, ... | `MalfunctionGenerator` | ✅ also on the graph view; the auto-pilot does not react to them (it waits on its fixed route) |
| demand generator | A generalization of the line and timetable generators to journeys and itineraries. | - | 🔴 not modelled: the journeys are fixed input to `PassengerEnv` |

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

The diagram shows what is implemented on this branch. Target design, not implemented yet, is marked: classes with
`<<target>>`, members with "target".

```mermaid
classDiagram
    direction LR
    note for PassengerEnv "Agents reflect passengers, waiting at an entry point or riding a vehicle. Actions: WAIT, HOP_ON, HOP_OFF. Plays the TravelWiseEnv: it steps the vehicles itself."
    note for RailEnv "Agents reflect trains/ships/etc. Positions are entry points: graph nodes or grid cell entry points (r,c,d). RailEnv and GraphRailEnv play the InfrastructureEnv."
    note for PassengerJourney "A journey: origin, given transfers and destination, each a waypoint group, with their times. A leg is two consecutive groups. No itinerary: the vehicles are picked at runtime."
    note for TransitionMap "Topology"
    note for Line "Services in Space. Waypoints are Waypoint objects on the grid, entry points on the graph. Passenger journeys have the same shape."
    note for Timetable "Services in Space and Time"
    note for StoppingPoint "aka. Haltepunkt. Today a stop is an entry point of a waypoint group."
    note for TimetablePredictor "Which stops each vehicle still serves, as if it ran alone and on time. No query view of all vehicles yet."
    RailEnv --> TransitionMap : rail
    RailEnv --> ResourceMap : resource_map
    RailEnv "1" --> "1.." EnvAgent : agents
    RailEnv --> ObservationBuilder : obs_builder
    RailEnv --> Rewards : rewards
    RailEnv --> MalfunctionGenerator : malfunction_generator
    ResourceMap ..> EntryPoint : maps to resources
    Timetable "1" --> "1" Line : per agent, aligned with its waypoints
    EnvAgent "1" --> "2.." LineFlexibleStoppingPoint : waypoints
    Line "1" --> "1.." LineFlexibleStoppingPoint
    LineFlexibleStoppingPoint "1" --> "1.." EntryPoint : alternatives
    StoppingPoint "1.." --> "1" Station
    StoppingPoint "1" --> "1" EntryPoint
    PassengerEnv --> RailEnv : rail_env, passengers use trains, ships etc.
    PassengerEnv --> ShortestPathPolicy : vehicle_policy, auto-pilot
    PassengerEnv --> PassengerObservationBuilder : obs_builder
    PassengerEnv --> PassengerRewards : rewards
    DefaultPassengerRewards --|> PassengerRewards
    PassengerEnv "1" --> "1.." PassengerJourney : passenger_journeys
    PassengerJourney "1" --> "2.." LineFlexibleStoppingPoint : waypoints
    ShortestPathPolicy ..|> Policy
    BoardForTargetPassengerPolicy ..|> Policy
    ShortestPathPolicy --> ShortestPathObservation : acts on
    BoardForTargetPassengerPolicy --> PassengerObservation : acts on
    BoardForTargetPassengerPolicy --> PassengerAction : returns
    PolicyRunner --> RailEnv
    PolicyRunner --> Policy
    PolicyRunner --> Trajectory : collects the run
    TravelWisePolicyRunner --> PassengerEnv
    TravelWisePolicyRunner --> Policy : passenger policy
    VehicleObservationBuilder --|> ObservationBuilder
    PassengerObservationBuilder --|> ObservationBuilder
    TimetablePredictor --|> PredictionBuilder
    VehicleObservationBuilder --> TimetablePredictor : predictor
    VehicleObservationBuilder --> ShortestPathObservation : builds
    PassengerObservationBuilder --> PassengerObservation : builds

    namespace Controller {
        class PolicyRunner {
            run()
        }
        class TravelWisePolicyRunner {
            reset()
            step()
            run()
        }
    }

    namespace Flatland {
        class RailEnv {
            rail: TransitionMap
            resource_map: ResourceMap
            agents: List[EnvAgent]
            obs_builder: ObservationBuilder
            rewards: Rewards
            malfunction_generator: MalfunctionGenerator
            effects_generator
            events_generator "target"
            step() "env step, called by controller"
            reset() "re-generate a new rail, line and timetable"
            register() "register for event topics, target"
            deregister() "de-register from event topics, target"
        }

        class TransitionMap {
            get_transitions(entry_point)
            apply_action_independent(action, entry_point)
            get_successor_entry_points(entry_point)
            get_predecessor_entry_points(entry_point)
            is_valid_entry_point(entry_point)
        }

        class ResourceMap {
            get_resource(from_entry_point, to_entry_point) "the occupancy unit for conflicts, e.g. a cell"
        }

        class EntryPoint {
            "a graph node or a grid cell entry point (r,c,d)"
        }

        class Station {
            <<target>>
            description: Any
        }

        class StoppingPoint {
            <<target>>
            description: Any
        }

        class Line {
            agent_waypoints: Dict[int, List[List[Waypoint]]]
            agent_speeds: List[float]
        }

        class LineFlexibleStoppingPoint {
            "a waypoint group: alternative entry points, one of them served"
        }

        class Timetable {
            earliest_departures: List[List[int]]
            latest_arrivals: List[List[int]]
            max_episode_steps: int
        }

        class EnvAgent {
            current_entry_point: Optional[EntryPoint]
            next_entry_point: Optional[EntryPoint]
            speed_counter: SpeedCounter "speed, and distance on the edge"
            waypoints: List[List[Waypoint]]
            waypoints_earliest_departure: List[int]
            waypoints_latest_arrival: List[int]
            next_stop() LineFlexibleStoppingPoint "target"
        }

        class Policy {
            <<interface>>
            act(observation)
            act_many(handles, observations, infos)
        }

        class ShortestPathPolicy {
        }

        class ShortestPathObservation {
            handle
            elapsed_steps
            done
            current_entry_point
            next_entry_point
            is_cell_exit
        }

        class ObservationBuilder {
            reset(env)
            get(handle)
            get_many(handles)
        }

        class PredictionBuilder {
            reset(env)
            get(handle)
        }

        class Rewards {
            step_reward()
            end_of_episode_reward()
        }

        class MalfunctionGenerator {
            "breakdowns, departure delays, ..."
        }

        class Trajectory {
            "actions, positions, rewards per step, env snapshots"
        }
    }
    namespace TravelWise {
        class PassengerEnv {
            reset()
            step()
            stop_index()
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
            elapsed_steps
            mode
            stop_index
            vehicles_to_board
            vehicle_stops_at_next_stop
        }
        class BoardForTargetPassengerPolicy {
        }
        class VehicleObservationBuilder {
        }
        class PassengerObservationBuilder {
        }
        class TimetablePredictor {
        }
        class PassengerRewards {
            <<abstract>>
            step_reward(journey, transition, elapsed_steps)
            end_of_episode_reward(journey, arrived, max_episode_steps)
        }
        class DefaultPassengerRewards {
        }
    }

```
