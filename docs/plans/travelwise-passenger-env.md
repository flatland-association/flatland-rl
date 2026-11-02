# Plan: TravelWise passenger env spike

PR [#528](https://github.com/flatland-association/flatland-rl/pull/528), branch
`feature/travelwise-passenger-overlway-spike`, based on `main` 3a66efe9. Status of 2026-11-02: all steps done.

## Goal

A technical spike of the TravelWise passenger env, in line with the
[data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md):

- The spike and its tests run on `GraphRailEnv` only, in the subpackage `flatland/envs/travelwise/` (phase 0).
- Each vehicle's remaining route is projected with optimistic times (phase 1).
- Passengers ride one vehicle from origin to target (phase 2), then with changes of trains at given stops (phase 5).
- Passengers observe the vehicles' positions and plans and decide themselves when to hop on and off; a controller
  holding the passenger policy steps `PassengerEnv`, which steps the vehicles with their own policy (phase 3).
- Passengers are rewarded for lateness (phase 4).

## Status

| Phase | ID | Commit message | Status |
|---|---|---|---|
| 0 | C0.1 | `refactor(envs): move the passenger env and its tests into the travelwise subpackage.` | 052da2a6 |
| 0 | C0.2 | `refactor(policies): inject the router into ShortestPathPolicy and let it serve intermediate stops.` | c62fffed |
| 0 | - | `refactor(graph): make the resource map of a graph converted from a grid public.` | bce9e133 |
| 0 | C0.3 | `refactor(envs): run the passenger env spike and its tests on GraphRailEnv only.` | f90b72da |
| 1 | C1.1 | `feat(envs): predict vehicles' remaining routes along their timetable in a new TimetablePredictor.` | d1283828 |
| 2 | C2.1 | `feat(envs): give passengers single-ride journeys with earliest departure and latest arrival.` | 14578ec9 |
| 3 | C3.0 | `refactor(policies): let ShortestPathPolicy act on a dedicated observation instead of the whole env.` | 6ebe3c72 |
| 3 | C3.1 | `feat(envs): let passengers hop on and off via actions, decided by a passenger policy on their observations.` | 2711d24d |
| 4 | C4.1 | `feat(envs): reward passengers for lateness and report per-passenger dones.` | 93a6de3f |
| 5 | C5.1 | `feat(envs): support passenger journeys with changes of trains at given stops.` | d2837293 |

## Design principles

- RL split (flatland-book, RLlib): an env's `step()` takes its agents' actions, updates its state and returns
  observations, rewards, dones and infos. It never runs the policy of its own agents; a policy (or a controller)
  takes the observations and issues the actions. Entities that are not the env's agents may be driven inside
  `step()` as part of the env's dynamics: `PassengerEnv` steps the vehicles with their own policy.
- The env is dull: it applies an action if it is physically feasible and ignores it otherwise. What to do - which
  vehicle to board, when to depart (the earliest departure), where to alight - is the policy's decision (Q23).
- Observations hold dynamic data only (positions, plans); they hold no position and no plan after `reset()`.
  Data shared by all agents is one immutable object referenced by every observation, not copied per agent.
- Static data (timetables, journeys, the map) is not part of the observations: it reaches the policies through the
  infos, as a reference to one immutable object (Q19). Observation builders, part of the env, read the env.

Reused concepts:

| Need | Reused concept | Instead of |
|---|---|---|
| Vehicle projection | `PredictionBuilder` (like `ShortestPathPredictorForRailEnv`: "as if no other agent is in the environment"), owned by the vehicles' observation builder, as `TreeObsForRailEnv` owns its predictor | a decorator around the rail env's observation builder; `PassengerEnv` owning the predictor |
| Passenger journey | `EnvAgent`'s `waypoints`, `waypoints_earliest_departure`, `waypoints_latest_arrival` and their `None` conventions | new fields `origin`, `target`, `via`, ... |
| Stop reached | exact entry point of a waypoint group, as in `BaseDefaultRewards` | resource (cell) matching |
| Vehicle's final stop | `EnvAgent.target_entry_point`, set by `handle_done_state()` | keeping the vehicles' entry points of the previous step |
| Observations | `ObservationBuilder` for vehicles and passengers | the whole env as observation (`FullEnvObservation`) |
| Static data | infos (RLlib: infos keyed by agent id; Flatland: `action_required` in the infos) | static data in every observation; policies reading the env |
| Policies | `Policy`/`act`, the `rail_env_policy.py`/`rail_env_policies.py` layout | - |
| Controller | the usual env/policy loop | the passenger policy inside `PassengerEnv.step()` |
| Passenger actions | an enum like `RailEnvActions` | - |
| Graph routing | `networkx` on `rail.g`, injected into `ShortestPathPolicy` and the predictor as a router | a hand-written search |

## Terms

Mapped to the TravelWise terms of flatland-book's
[TravelWise page](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise.md) and
[data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md)
(no glossary there; terms from its environments table, prose and diagram notes). "-": no TravelWise term.

| Term (code) | TravelWise term | Meaning here, and what it is not |
|---|---|---|
| vehicle (`GraphRailEnv` agent, `handle`) | vehicle; agent of the infrastructure (ground) level | A train running on the rail env, on auto-pilot, with no notion of passengers. The tests also say "train"; not a passenger's ride (a leg). |
| rail env, vehicle env (`GraphRailEnv`) | `InfrastructureEnv` (a generalized `RailEnv`) | The vehicles' env on the network (graph) view. Rail only: no multimodal edges, no walkways. |
| passenger (`PassengerEnv` agent, `handle`) | passenger (a group of N people) | One agent per passenger; group size and capacities are not modelled. |
| passenger env (`PassengerEnv`) | `PassengerEnv` (passenger level) | Steps the passengers on top of a rail env it holds. Not a subclass of the rail env, as the data model draws it; no `TravelWiseEnv` combining both. |
| entry point (graph node), resource | configuration (aka node, grid cell entry point) | Where a vehicle or a waiting passenger is; a resource is what two vehicles cannot occupy at once (a cell). |
| stop (one entry point) | `StoppingPoint` (Haltepunkt) | A station track where a vehicle stops. Not a station: stations are not modelled. |
| waypoint group (`waypoints[i]`) | `LineFlexibleStoppingPoint` | Alternative stops of which a vehicle uses one, or at which a passenger boards, changes or arrives. |
| line and timetable (`EnvAgent.waypoints`, `Timetable`) | line; timetable | A vehicle's stops, and its earliest departure and latest arrival per stop. |
| journey (`PassengerJourney`) | journey (origin and destination); partly initial itinerary | Origin, given changes and target, with earliest departure and latest arrival per group. More than TravelWise's journey (the changes and times), less than an itinerary (no vehicles chosen). |
| change (intermediate group of a journey) | transfer | Where the passenger gets off and waits for the next vehicle. Given, not planned; no minimum transfer time. Its meaning may change with the query view (see follow-ups). |
| ride (two consecutive groups of a journey) | leg | Part of a journey on one vehicle. |
| itinerary | initial itinerary | Not modelled: the passenger policy picks the vehicle at each stop at runtime. |
| hop on, hop off (`PassengerAction.HOP_ON`, `HOP_OFF`) | hop down on, hop back up | Board a vehicle at the passenger's resource; get off at the vehicle's current entry point. |
| wait (`PassengerAction.WAIT`) | move to edge (the passenger's only action) | Stay where it is. Passengers do not walk: no walkways. |
| passenger mode (`passenger_mode`, `PassengerMode`) | - | Where a passenger is: waiting at a resource, or riding a vehicle. Not TravelWise's mode of transport. |
| next stop (`PassengerEnv.next_stop()`) | - | The passenger's next change or its target. |
| plan (tuple of `ProjectedEntryPoint`) | - | A vehicle's predicted remaining route and stop times, as if it ran alone and on time. Not its timetable, not its policy's intent. |
| query view | global observation of the graph and timetables | All vehicles' future stops, to plan connections. Not built (follow-up). |
| vehicle policy, auto-pilot (`ShortestPathPolicy`) | auto-pilot; `FlatlandPolicy` | Drives the vehicles inside `PassengerEnv.step()`. |
| passenger policy (`BoardForTargetPassengerPolicy`) | `PassengerPolicy` | Decides the passengers' actions. Not an itinerary planner yet, which TravelWise expects it to become. |
| controller (the test loop) | `TravelWisePolicyRunner` | Holds the passenger policy and steps `PassengerEnv`. No library class yet (follow-up). |
| static data (`infos["static"]`) | - | What doesn't change during an episode (map, timetables, journeys), handed to the policies in the infos. |
| milestone, event | milestone; event | Not modelled in this PR. |

## Components

Added and modified by this PR (🟩 added, 🟨 modified):

| Component | Responsibility | Change |
|---|---|---|
| `PassengerEnv` | Simulates passengers travelling on the trains of a rail env: applies their hop-on and hop-off decisions, carries riding passengers along, and rewards punctual arrival. Not for deciding what passengers do (that is the passenger policy), nor for driving the trains (it delegates them to the vehicle policy). | 🟩 added: passengers ride the trains of a rail env on the network (graph) view, decide themselves when to board and get off, can change vehicles at given stops, and are rewarded for being on time. |
| `PassengerJourney` | A passenger's travel request: where it starts, where it changes vehicles, where it ends, and when. Not a plan of which vehicles to take. | 🟩 added: passengers get an origin, changes and a target with earliest departures and latest arrivals, written like a train's timetable. |
| `PassengerAction` | What a passenger can do in a step: wait, board a vehicle, or get off. Not whether doing so is wise. | 🟩 added: boarding and getting off are passenger actions instead of happening automatically. |
| `PassengerObservation` (`PassengerObservationBuilder`) | What a passenger sees in a step: its own situation and the vehicles it could take towards its next stop. Not the static journey or timetable data, which come with the infos. | 🟩 added: passengers can see which vehicles will take them to their next stop, and when. |
| `BoardForTargetPassengerPolicy` | A simple passenger behaviour: board the first suitable vehicle towards the next stop and get off there. Not a journey planner: it does not choose connections. | 🟩 added: a ready-made passenger behaviour to run the env with, and a baseline to compare others to. |
| `VehicleObservation` (`VehicleObservationBuilder`) | What each vehicle (and, through it, every passenger) knows of all vehicles in a step: where they are and where and when they will stop. Not the timetable itself. | 🟩 added: every vehicle's position and announced stops are visible to all, shared rather than copied per agent. |
| `TimetablePredictor` (`UnilateralTimetablePredictor`) | Announces each vehicle's remaining route and stop times, assuming it runs undisturbed and on time. Not a forecast of delays or of conflicts between vehicles. | 🟩 added: vehicles announce where they will stop and when, so passengers can plan their boarding. |
| `ShortestPathObservation` (`ShortestPathObservationBuilder`) | What the auto-pilot needs to know of its own vehicle in a step. Not the map or the timetable, which come with the infos. | 🟩 added: the auto-pilot can run on a small observation instead of the whole env. |
| `RailEnvStaticData` (`AgentStaticData`) | The rail env's data that doesn't change during an episode (map, routes, timetables), handed to policies once instead of in every observation. Not dynamic state. | 🟩 added: policies get the map and timetables through the infos, without reading the env. |
| `ShortestPathPolicy` | The auto-pilot: drives each vehicle along a shortest route through its stops, keeping to its timetable at intermediate stops on request. Not for avoiding conflicts with other vehicles. | 🟨 modified: trains can now wait at intermediate stops until their departure, run on the graph, and use the small observation; existing callers keep working unchanged. |
| Routers (`greedy_grid_route_through_waypoints`, `greedy_graph_route_through_waypoints`) | Find a route through a vehicle's stops on the grid or the graph, shared by the auto-pilot and the predictor so both agree. Not optimal over all alternatives (greedy per stop). | 🟩 added: routing is pluggable, and what the vehicles announce matches how they drive. |
| `GraphTransitionMap` (`grid_to_digraph`, `grid_to_resource_map`) | Converts a grid map into the network (graph) view. Not a native graph builder. | 🟨 modified: a graph and its resources can be built from a grid map without creating a `RailEnv` first. |

## Phase 0: graph-only spike (done)

- Goal: the passenger simulation runs on the network (graph) view only, and trains run exactly on their timetable,
  including waiting at intermediate stops until their departure.
- `ShortestPathPolicy` takes an injected router (`greedy_grid_route_through_waypoints()` by default,
  `greedy_graph_route_through_waypoints()` for graphs, `networkx` over `rail.g`) and a flag
  `serve_intermediate_stops` (default off, the previous behaviour): with it, the agent halts at each intermediate
  stop until its earliest departure. TODO in the graph router: replace it by `get_k_shortest_paths()` once that is
  generalized to graphs.
- The corridor graph is built from a `GridTransitionMap` with `GraphTransitionMap.grid_to_digraph()` and
  `grid_to_resource_map()`, without `RailEnv` (TODO: build the graph natively).
- `PassengerEnv.clone_from()` is removed: no caller, and `GraphRailEnv` has no `clone_from()`.
- Graph edges count as length 1 until [#542](https://github.com/flatland-association/flatland-rl/pull/542).

## Phase 1: per-vehicle projection (done)

- Goal: each vehicle announces where it will be and when, assuming it runs undisturbed and on time.
- `TimetablePredictor` (interface, overrides `get()` only) and `UnilateralTimetablePredictor` in
  `flatland/envs/travelwise/timetable_predictor.py`: per vehicle, its remaining route as a list of
  `ProjectedEntryPoint(entry_point, step, stop)` - `stop` the index of the waypoint group if the vehicle stops
  there. `get()` advances each vehicle's waypoint progress (exact entry points, as `BaseDefaultRewards`), then
  routes from the current (or initial) entry point through the remaining groups with the injected router.
- Times: off map, the vehicle enters at `max(earliest departure, elapsed + 1)`; each move takes
  `ceil((1 - distance) / max_speed)` steps; at an intermediate stop it moves again at
  `max(arrival + 1, earliest departure)`. Exact for a single vehicle alone on the map, otherwise optimistic.
- Not the query view: a multi-vehicle projection of all vehicles' future stops comes later.

## Phase 2: single-ride journeys (done)

- Goal: each passenger has a journey from an origin to a destination, with an earliest departure and a latest
  arrival, on a single train that stops at the destination.
- `PassengerJourney(waypoints, waypoints_earliest_departure, waypoints_latest_arrival)`, the same fields and
  conventions as `EnvAgent`; exactly two groups (origin with one entry point, target).
- A vehicle serves a passenger if one of its future stop groups shares an entry point with the passenger's next
  stop (Q10). The vehicle stops at one entry point of the group, chosen by its policy - possibly outside the
  passenger's next stop: then the passenger misses it and stays on board up to the vehicle's final stop.
- Boarding and alighting were automatic in this phase; phase 3 replaces them by actions.
- A vehicle reaching its target is removed in that step: riding passengers are forced off at
  `EnvAgent.target_entry_point`.

## Phase 3: passenger decisions (done)

### Goal

- Passengers decide themselves when to board a vehicle and when to leave it, based on what they observe of the
  vehicles.
- The vehicles' policy acts on a dedicated observation instead of the whole env (C3.0).

### C3.0: `ShortestPathPolicy` on a dedicated observation

Today `ShortestPathPolicy` needs `FullEnvObservation` (the whole env). It reads the transition map (`env.rail`),
`env._elapsed_steps` and, from `env.agents[handle]`: `handle`, `state` (done only), `current_entry_point`,
`next_entry_point`, `speed_counter.is_cell_exit()`, `initial_entry_point`, `waypoints` and
`waypoints_earliest_departure`.

- `ShortestPathObservation` (frozen dataclass), the dynamic part: `handle`, `elapsed_steps`, `done`,
  `current_entry_point`, `next_entry_point`, `is_cell_exit` - `elapsed_steps` only for now (Q18). After `reset()` it
  has no position.
- The static part - the map, and per vehicle `initial_entry_point`, `waypoints`, `waypoints_earliest_departure` -
  reaches the policy through the infos (Q19), added by the runner (Q20): `PassengerEnv` for the vehicles, the test
  loops in the tests.
- `ShortestPathObservationBuilder(ObservationBuilder[AbstractRailEnv, ShortestPathObservation])` builds it.
- `ShortestPathPolicy` reads the observation and the static data instead of `env`/`agent`.
- Backwards compatible: with `FullEnvObservation` (the env as observation), the policy derives both from the env, so
  existing callers keep working (Q22).
- Tests: the existing ones unchanged (compatibility path), plus the policy with the new builder on grid and graph;
  the travelwise tests switch to the new builder.

### C3.1: actions, observations and policy

Actions and validation:

- Action per passenger: `(PassengerAction, vehicle)`, `PassengerAction` one of `WAIT`, `HOP_ON`, `HOP_OFF`
  (`flatland/envs/travelwise/passenger_env_action.py`); the vehicle handle only with `HOP_ON`.
- The env applies an action if it is physically feasible and ignores it otherwise. `HOP_ON`: the passenger waits
  and the vehicle is on the map at the passenger's resource - checked on the state the passenger observed.
  `HOP_OFF`: while riding, at the vehicle's current entry point. Whether the vehicle serves the passenger and
  whether the passenger's earliest departure has passed are the policy's decisions (Q23).
- The forced hop-off at a vehicle's final stop stays (phase 2).
- Arrived: alighted at an entry point of the target; the passenger is done, and its further actions are ignored.

Observations:

- `VehicleObservationBuilder` (`flatland/envs/travelwise/vehicle_observations.py`, for now) is the rail env's observation builder: it owns
  the predictor and builds, per vehicle, its `ShortestPathObservation` plus a reference to the shared, immutable
  positions and plans of all vehicles (Q12, Q14) - a plan is a tuple of `ProjectedEntryPoint`s.
- `PassengerObservation` (`flatland/envs/travelwise/passenger_env_observations.py`): the elapsed steps (for now, as
  for the vehicles, Q18), the passenger's mode and next stop, the vehicles it can hop on now that serve it, with their future stops serving its next stop, and whether
  the vehicle it rides stops at its next stop now. Built by `PassengerObservationBuilder` from the vehicle
  observations `PassengerEnv` keeps, and from the journeys and timetables in the env. No position and no plan after
  `reset()`: passengers and vehicles are off the map (Q11, Q15).

Policy:

- `BoardForTargetPassengerPolicy` (`flatland/envs/travelwise/passenger_env_policies.py`) acts on the observation,
  with the passenger's journey from the static data in the infos:
  - riding, its vehicle stopping at its next stop: `HOP_OFF`;
  - waiting, boarding not before its earliest departure, with vehicles to board: `HOP_ON` - first a vehicle stopping
    at an entry point of the next stop, then the one stopping there soonest. The action applies in the step after the
    one observed, so it boards if `elapsed_steps + 1 >= earliest departure` - the rule `ShortestPathPolicy` departs
    a vehicle by;
  - otherwise `WAIT`.

### Controller and step

The controller (the data model's `TravelWisePolicyRunner`, for now the test loop) holds the passenger policy and
steps only `PassengerEnv`, with the passengers' actions. `PassengerEnv` holds the rail env and the vehicle policy
(the auto-pilot), and is the runner of the vehicles: it keeps their last observations and infos, the vehicle policy
acts on them in the next step, and the passengers' observations are built from them - each state is observed once.

Within `PassengerEnv.step()`, the passengers' actions are applied first, on the state they observed at the end of
the previous step; then the vehicles move, and the riding passengers with them. So passengers and vehicles both
decide on the same state. A passenger boards, or alights at an intermediate stop, in the step after it observed
the vehicle there - e.g. T0 stands at A only in step 2: a passenger observing it there boards in step 3, as T0
moves on. A passenger arrives in the step its `HOP_OFF` is applied; the forced hop-off at a vehicle's final stop
happens in the step the vehicle arrives.

```mermaid
sequenceDiagram
    participant C as TravelWisePolicyRunner (controller, for now the test loop)
    participant PP as BoardForTargetPassengerPolicy (passenger policy)
    participant P as PassengerEnv (passenger env)
    participant VP as ShortestPathPolicy (vehicle policy, auto-pilot)
    participant R as GraphRailEnv (rail env, vehicle env)
    participant VO as VehicleObservationBuilder (vehicle env's obs builder)
    participant O as PassengerObservationBuilder
    C->>P: reset()
    P->>R: reset()
    R->>VO: reset(rail env), get_many(vehicle handles)
    VO->>VO: predictor.reset()
    VO-->>R: vehicle observations: empty - no positions, no plans yet
    R-->>P: vehicle observations, infos (static data)
    P->>P: keep vehicle observations and infos, passengers off map
    P->>O: reset(passenger env), get_many(passenger handles)
    O-->>P: passenger observations: empty
    P-->>C: passenger observations, infos (static data)
    loop step t = 1, 2, ... until passenger env done
        Note over C,O: passenger and vehicle observations and infos of state t-1
        C->>PP: act_many(passenger handles, passenger observations, infos)
        PP-->>C: passenger actions
        C->>P: step(passenger actions)
        P->>P: apply passenger actions (invalid ones ignored): HOP_ON at the vehicle's entry point, HOP_OFF
        opt auto-pilot
            P->>VP: act_many(vehicle handles, vehicle observations, infos)
            VP-->>P: vehicle actions
            P->>R: step(vehicle actions)
            R->>VO: get_many(vehicle handles)
            VO->>VO: predictor.get() - vehicles' plans in state t
            VO-->>R: vehicle observations: own state, all vehicles' positions and plans
            R-->>P: vehicle observations, dones, infos (static data)
        end
        P->>P: keep vehicle observations and infos, passengers show up at their origin (step 1), riding passengers follow their vehicles,<br/>forced hop-off at a vehicle's final stop, arrivals
        P->>O: get_many(passenger handles)
        O->>O: read positions and plans from vehicle observations, timetables and journeys from the env
        O-->>P: passenger observations: own state, vehicles' positions and plans
        P-->>C: passenger observations, rewards, dones, infos (static data)
    end
```

## Phase 4: lateness rewards and dones (done)

- Goal: passengers are rewarded the earlier they arrive before their latest arrival, penalized for arriving late or
  not at all; each passenger's journey is done once it arrives.
- Rewards per passenger, in the step they occur, otherwise 0. They can be positive (early arrival), unlike
  Flatland's `DefaultRewards` (`min(latest_arrival - arrival_time, 0)`, never positive): good enough for the
  spike, to be revisited (follow-up):
  - at its arrival step: `waypoints_latest_arrival[-1] - arrival_time` (late negative);
  - at boarding before the earliest departure of the group it waits at: `-(earliest departure - boarding step)`,
    the boarding step being the step the `HOP_ON` is applied (Q23);
  - when the rail env is done and it has not arrived: `-max_episode_steps` (Q6).
- `step()` returns the passenger observations, rewards per passenger, dones per passenger (arrived) plus
  `'__all__'` (rail env done), and the infos.

## Phase 5: journeys with changes of trains (done)

- Goal: passengers can change trains at given stops along their journey.
- `PassengerJourney.waypoints` may have intermediate groups, the stops where the passenger changes; each pair of
  consecutive groups is one ride. `next_stop(p)` is the first group not yet reached; boarding, alighting, the
  forced hop-off and the missed stop apply per ride.
- At a change the passenger alights, waits and boards the next vehicle serving it; respecting the group's earliest
  departure is the passenger policy's decision (Q23). No minimum transfer time (TODO).
- Arrived and rewards: at the target only.
- Tests: passenger 0 of the shuttle tests becomes A -> B -> C and arrives again.

## Limitations (`PassengerEnv`'s docstring)

- The changes are given, not planned: no search for connections (comes with the query view). No minimum transfer
  time (TODO).
- A passenger may board where the vehicle does not stop, and hop off at any step (TODOs).
- Missed stop at another alternative of a stop group: not covered by a test, as every group on the corridor has one
  entry point (TODO).
- The per-vehicle projection is optimistic; no query view yet.
- Stops are station tracks (entry points), not stations.
- Graph edges count as length 1 ([#542](https://github.com/flatland-association/flatland-rl/pull/542)).

## Test scenario (on the graph)

Corridor A-B-C in `tests/flatland/envs/travelwise/`. Vehicles: T0 A -> B -> C, stopping at B until its earliest
departure there; T1 C -> B; T2 a later A -> B. Passengers:

- A -> C rides T0 through B;
- A -> B on T0 hops off at the intermediate stop B, late (negative reward in phase 4);
- B -> C boards T0 at B;
- A -> B with earliest departure 4, after T0 has left A (boarding T0 at A would be in step 3), takes T2;
- C -> B on T1: forced hop-off at the vehicle's final stop;
- C -> A: no vehicle stops at A heading west, so it never arrives.

Tests per phase: predictor unit tests and a single-vehicle exactness test (speeds 1, 1/2, 1/3; departure before and
after the vehicle is on the map); mode changes and arrivals (phase 2); decisions via the passenger policy and the
observations (phase 3); rewards and dones (phase 4); changes of trains in the shuttle tests (phase 5).

## Assumptions

Not checked in code; they hold by construction of the tests and the env's setup:

- Vehicles have no capacity limit: any number of passengers can ride one vehicle.
- Static data (the map, the timetables, the journeys) does not change during an episode; the objects handed out in
  the infos are read-only by convention.
- The vehicles follow the routes the predictor computes: both use the same router (Q1, Q14). A vehicle policy with
  another router makes the plans wrong.
- Waypoint groups are disjoint and visited once per vehicle and per journey (see the follow-ups).
- The rail env is a `GraphRailEnv`: the spike and its tests do not run on the grid view.

## New patterns

Compared with `main`; the reused concepts are listed under the design principles.

| New | Reuses | Defined in |
|---|---|---|
| Static data handed to policies in the infos (`infos["static"]`, passed to `act_many()`), not in the observations | RLlib's infos, Flatland's `action_required` in the infos | Design principles, Q19, Q20 |
| An env that is the runner of another env's agents: `PassengerEnv` steps the vehicles with their policy inside its `step()` | the usual env/policy loop | Controller and step |
| Data shared by all agents as one immutable object referenced by every observation | - | Q12 |
| A policy and a predictor with an injected router, so both pick the same entry points | - | Phase 0, Q1 |

Not new on purpose: the rail env's observations, infos and rewards, the trajectory format, `PolicyRunner` and the
dependencies (`networkx` is already one) stay as they are.

## Breaking changes

None for callers: `ShortestPathPolicy()` keeps its defaults, and the env as observation (`FullEnvObservation`) keeps
working (Q22). Everything under `flatland/envs/travelwise/` is new. Changes visible only to subclasses or type checks:

- `ShortestPathPolicy` no longer subclasses `RailEnvPolicy`, which binds the env to `RailEnv`: it is a `Policy` over
  `AbstractRailEnv`, to run on `GraphRailEnv` too. An `isinstance(policy, RailEnvPolicy)` check fails; there is none
  in flatland-rl or flatland-baselines.
- `ShortestPathPolicy`'s private `_act()` and `_update_agent()` take the observation and the static data instead of
  the env and the agent: a subclass overriding them breaks; none is known.
- `GraphRailEnv._grid_resource_map()` (private) is replaced by the public `GraphTransitionMap.grid_to_resource_map()`.

## Follow-ups (not in this PR)

- Board only where the vehicle stops; hop off only at stops; missed stop at another alternative.
- Minimum transfer time (phase 5).
- Query view: a multi-vehicle projection of all vehicles' future stops, replacing the per-vehicle predictor.
  With it, passengers plan their connections, and a journey may go back to only an origin and a target (as
  TravelWise's journey). If journeys keep intermediate groups, fix what they mean: hints where to change, or stops
  the passenger must make for their own sake (an activity such as shopping or leisure, maybe with a dwell time).
- `get_k_shortest_paths()` generalized to graphs, replacing the `networkx` routing.
- Build the test corridor as a graph natively.
- `TravelWisePolicyRunner` as a library class instead of the test loop.
- Rewards: align with Flatland's `DefaultRewards` (no bonus for arriving early) or keep the bonus (phase 4).
- Waypoint groups are tacitly assumed to be disjoint and visited once (rail env target check, rewards,
  `BoardForTargetPassengerPolicy`, the predictor): validate or support repeated visits, in a separate PR.

## Decisions

- Q1 Routing: a shared greedy `networkx` router over `rail.g` (nearest alternative per group), injected into
  `ShortestPathPolicy` and the predictor, so both pick the same entry points.
- Q2 Existing shuttle tests keep their setup; passenger 0 (A -> C) never arrives with single rides, and becomes
  A -> B -> C in phase 5.
- Q3 Boarding only from the passenger's earliest departure on - since Q23 the passenger policy's decision, not
  checked by the env.
- Q5 Layout: `flatland/envs/travelwise/` with `passenger_env.py`, `passenger_env_action.py`,
  `passenger_env_observations.py`, `passenger_env_policies.py`, `timetable_predictor.py`, following
  `rail_env.py`/`rail_env_action.py`/`rail_env_policies.py`; no `PassengerEnvPolicy` base class.
- Q6 An arrival in the step the rail env finishes is rewarded as an arrival.
- Q7 The test timetable generator takes a departure time per stop.
- Q8 The late passenger is the A -> B passenger on T0.
- Q9 The passenger without a vehicle is C -> A.
- Q10 Boarding rule: a vehicle's future stop group shares an entry point with the passenger's next stop.
- Q11 Observations are empty after `reset()` and hold only dynamic data; static data via the infos (Q19).
- Q12 Data shared by all agents (positions and plans) is one immutable object referenced by every observation: with
  n vehicles, a copy per vehicle would be n * n plans.
- Q13 The plan is the projection, per entry point with stop marks; the timetable is static data.
- Q14 The plans are the env's prediction (same router as the vehicle policy), not the policy's intent.
- Q15 Passengers are off the map after `reset()`.
- Q16 The vehicles' dedicated observation comes first, as C3.0.
- Q17 `VehicleObservationBuilder` lives in `flatland/envs/travelwise/` for now.
- Q18 The vehicle's state (`done`, entry points, `is_cell_exit`) is dynamic: it goes into the observation. The
  elapsed steps belong into the infos, but the rail env's infos do not have them, and adding them is a rail env
  change: for now they go into the observation too (`ShortestPathObservation.elapsed_steps`). TODO: move them to the
  infos with the rail env change of Q20.
- Q19 Static data reaches the policies through the infos, as a reference to one immutable object; runners pass the
  infos to `act_many()` through `**kwargs`. Follows RLlib (infos keyed by agent id, `"__common__"` for data shared
  by all agents; the old API's `Policy.compute_actions(..., info_batch=...)`) and Flatland's own decision data in the
  infos (`action_required`); gymnasium describes infos as auxiliary.
- Q20 Who puts the vehicles' static data into the infos: Flatland's infos are `infos[key][handle]` (RLlib:
  `infos[handle][key]`), and `PolicyRunner` persists `{key: infos[key][handle]}` per agent and step into the
  trajectory - static data in the rail env's infos would be persisted every step, and the map does not serialize.
  Decided: the rail env's infos stay unchanged in this PR. `PassengerEnv`, the vehicles' runner, adds the vehicles'
  static data to the infos it passes to the vehicle policy (`infos["static"]`, the role of an RLlib env-to-module
  connector), and hands out the passengers' static data in its own infos. Other runners (`PolicyRunner`) keep
  using `ShortestPathPolicy` with `FullEnvObservation` until the rail env hands out its static data (TODO, a core
  change with trajectory persistence skipping it).
- Q22 Backwards compatibility of `ShortestPathPolicy` with `FullEnvObservation` (the env as observation): one
  helper - if an observation is the env, `act_many()` first derives the `ShortestPathObservation` and the static data
  from it, then runs the same code as for the new observation; one code path for the routing and stop logic.
- Q23 Boarding before the earliest departure: the env does not enforce it (the env is dull, the policy decides); the
  rewards penalize it per step boarded too early, `earliest departure - boarding step`, on the same scale as the
  lateness reward, so a policy gains nothing by ignoring the constraint.
