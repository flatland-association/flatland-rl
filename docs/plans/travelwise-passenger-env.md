# Plan: TravelWise passenger env spike

PR [#528](https://github.com/flatland-association/flatland-rl/pull/528), branch
`feature/travelwise-passenger-overlway-spike`, based on `main` 3a66efe9. Status of 2026-11-03: phases 0-5 done, phase 6
(refactorings and cleanup) planned.

## Goal

A technical spike of the TravelWise passenger env, in line with the
[data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md):

- The spike and its tests run on `GraphRailEnv` only, in the subpackage `flatland/envs/travelwise/` (phase 0).
- Each vehicle's remaining route is projected with optimistic times (phase 1).
- Passengers ride one vehicle from origin to target (phase 2), then with changes of trains at given stops (phase 5).
- Passengers observe the vehicles' positions and plans and decide themselves when to hop on and off; a controller
  holding the passenger policy steps `PassengerEnv`, which steps the vehicles with their own policy (phase 3).
- Passengers are rewarded for lateness (phase 4).
- Refactorings and cleanup: a runner instead of the test loop, and the code review findings fixed (phase 6).

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
| 6 | C6.1 | `feat(envs): add a TravelWisePolicyRunner running a passenger env episode with a passenger policy.` | b5324b2b |
| 6 | C6.2 | `fix(policies): reset ShortestPathPolicy's routes when a new episode starts.` | d19bbc88 |
| 6 | C6.3 | `fix(policies): fail when ShortestPathPolicy cannot route through all waypoint groups.` | 673953fd |
| 6 | C6.4 | `fix(envs): let the passenger policy read the earliest departure of the group it waits at from its observation.` | 6835da3e |
| 6 | C6.5 | `fix(envs): warn on and ignore malformed passenger actions and unknown vehicle handles.` | open |
| 6 | C6.6 | `refactor(policies): extract the stop decision and the first projected step into their own methods.` | open |
| 6 | C6.7 | `refactor(envs): tighten the type hints of the passenger env.` | open |
| 6 | C6.8 | `perf(envs): compute a vehicle's stops serving a passenger once per observation.` | open |

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
(no glossary there; terms from its environments table, prose and diagram notes). "-": no TravelWise term. Both pages
are copied into this repo, [`docs/travelwise.md`](../travelwise.md) and
[`docs/travelwise/data_model.md`](../travelwise/data_model.md), and updated to the decisions of this plan.

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
| route (list of `EntryPointT`, from a `Router`) | - | Every entry point a vehicle passes, in order (on the grid each cell entered, with its direction; on the graph each node), from its initial entry point through one alternative per waypoint group. Neighbouring entries are connected by a transition. The waypoints are the few entries the stop indices point to. Found greedily per group, so not the shortest over all alternatives. Not its line (the stops only), not a passenger's itinerary. |
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
| controller, runner (`TravelWisePolicyRunner`) | `TravelWisePolicyRunner` | Holds the passenger policy and steps `PassengerEnv`; the vehicles are stepped by `PassengerEnv`. A library class since C6.1. |
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
| `TravelWisePolicyRunner` | Runs a passenger episode: resets the passenger env, then steps it until it is done, each time with the passenger policy's actions on the previous observations and infos. Not for stepping the vehicles (`PassengerEnv` does) and not for recording trajectories. | 🟩 added: a passenger episode runs in one call, without writing the loop. |
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
  for the vehicles, Q18), the passenger's mode and stop index (the journey's group it is at or last left; its next
  stop is the group after it, read from the journey in the static data, Q11), the vehicles it can hop on now that serve it, with their future stops serving its next stop, and whether
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

The controller (`TravelWisePolicyRunner`, the data model's name, C6.1) holds the passenger policy and
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
    participant C as TravelWisePolicyRunner (controller)
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

## Phase 6: refactorings and cleanup (planned)

Goal: clean up the spike's code before merging - a runner instead of the test loop (C6.1), and the code review
findings 40-46 fixed (C6.2-C6.8). No new feature, and no change to the test traces. Each step is one commit, green
on its own, with its tests.

### C6.1: the runner as a library class (done)

- Goal: a library user runs a passenger episode in one call instead of copying the test loop; the tests use the
  same runner (finding 16).
- Today the controller is the test helper `_run()` in `tests/flatland/envs/travelwise/test_passenger_env_scenario.py`.
  `PolicyRunner` is not reused: it is bound to `RailEnv` and trajectories (it loads the env from a trajectory and
  records actions, positions and rewards per train), and it does not pass the infos (static data, Q19) to the
  policy. Recording passenger runs in trajectories stays a follow-up (finding 22).
- `TravelWisePolicyRunner(passenger_env, passenger_policy, callbacks=None)` in
  `flatland/envs/travelwise/travelwise_policy_runner.py` (Q24):
  - `reset()`: resets the passenger env, keeps the observations and infos and returns them;
  - `step()`: the passenger policy's actions on the kept observations and infos, then `passenger_env.step()`; keeps
    the new observations and infos, sums the rewards per passenger (`total_rewards`), returns
    `(observations, rewards, dones, infos)`;
  - `run()`: `reset()`, then `step()` until `passenger_env.done`; calls `FlatlandCallbacks`' `on_episode_start`,
    `on_episode_step` and `on_episode_end` with `env=passenger_env`, as `PolicyRunner` does; returns
    `total_rewards`.
- Unchanged: `PassengerEnv` steps the vehicles with the vehicle policy (Controller and step, Q25); no trajectories.
- Tests: the test loops (`_run()` in the scenario test, the changes test in the spike test) step the runner
  (`runner.reset()`/`runner.step()`), still recording mode changes and rewards per step, with the traces unchanged; a
  new test checks the rewards `run()` returns and the order of the callbacks. A second `run()` on the same passenger
  env needs C6.2: the vehicle policy keeps its routes across episodes (finding 40); C6.2's test covers it.
- Docs: `PassengerEnv`'s docstring loop example uses the runner; the controller in the terms, the components and the
  sequence diagram is the runner.

### C6.2-C6.8: code review findings

| ID | Finding | Change | Tests |
|---|---|---|---|
| C6.2 | 40 | `ShortestPathPolicy` drops its routes when a new episode starts: at the first observation with `elapsed_steps == 0` (or the env as observation at step 0). Also fixes the failure on `main` for agents not done in the first episode. | The policy reused for two episodes on different lines; both run to the end. `TravelWisePolicyRunner.run()` twice on the same passenger env returns the same total rewards. |
| C6.3 | 43 | `ShortestPathPolicy` raises when the router routes fewer waypoint groups than given, as `main` asserted. The routers keep returning the groups they reached; the predictor keeps ending its plan there. | An unreachable group raises in `act_many()`. |
| C6.4 | 41 | `PassengerObservation` gets `stop_index` (`PassengerEnv.stop_index()`), the index of the journey's group the passenger is at or last left, in place of `next_stop`: the observation keeps only dynamic data (Q11), the groups and their times are static data in the infos. `BoardForTargetPassengerPolicy` reads the earliest departure and the next stop from the journey at that index, instead of searching the journey by `next_stop`. Repeated visits stay tracked in [#557](https://github.com/flatland-association/flatland-rl/issues/557). | The observation test asserts the field; the scenario traces stay unchanged. A policy test on a journey A -> B -> A boards at B not before B's earliest departure (with the search by `next_stop`, it boards too early). |
| C6.5 | 42 | `PassengerEnv` ignores, with a warning, an action that is not an `(action, handle)` pair, and a `HOP_ON` whose vehicle handle is not one of the rail env's. `BoardForTargetPassengerPolicy` never sends such actions (it boards only vehicles from its observation); they come from other policies, e.g. an RL policy whose action space does not bound the vehicle handle, so the env warns instead of hiding the bug. The rail env's handles are its agents' indices `0, ..., n - 1` (`AbstractRailEnv.get_agent_handles()`), so the handle is checked by range (an integer, numpy integers included, with `0 <= handle < get_num_agents()`): a negative handle no longer boards a vehicle counted from the end, one out of range no longer raises. Infeasible but well-formed actions (e.g. hopping on a vehicle elsewhere) stay ignored without a warning. | A passenger waiting at a vehicle gets a warning for scripted actions with a negative handle, one out of range, one not an integer, a negative numpy integer, and a bare `PassengerAction`, and boards one step later by its policy. The other tests run without the warning. |
| C6.6 | 46 | `ShortestPathPolicy._act()`: the stop decision at intermediate stops moves into its own method. `UnilateralTimetablePredictor._predict()`: the first projected step (off the map, halted, moving) moves into its own method. No behaviour change. | The existing policy and predictor tests. |
| C6.7 | 44 | `PassengerObservation.mode` is a `PassengerMode`; `BoardForTargetPassengerPolicy.act()` requires `journey`; `PassengerEnv.vehicle_observations` is a `Dict[int, VehicleObservation]`. | flake8 and the existing tests. |
| C6.8 | 45 | `PassengerObservationBuilder.get()` computes `future_stops_serving()` once per vehicle and keeps the non-empty results; `serves()` goes if unused. | The existing observation tests. |

Not in phase 6: the spike's test module keeps its name until the design is stable (finding 15); the other open
findings are follow-ups.

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
observations (phase 3); rewards and dones (phase 4); changes of trains in the shuttle tests (phase 5); the runner and
the fixes of phase 6.

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
- Rewards: align with Flatland's `DefaultRewards` (no bonus for arriving early) or keep the bonus (phase 4).
- Waypoint groups are tacitly assumed to be disjoint and visited once (rail env target check, rewards,
  `BoardForTargetPassengerPolicy`, the predictor): validate or support repeated visits, in a separate PR
  ([#557](https://github.com/flatland-association/flatland-rl/issues/557)).
- An RL wrapper for the passenger env (e.g. for RLlib or gymnasium): its action space bounds the vehicle handle of
  `HOP_ON` to the rail env's `0, ..., n - 1`, so an RL policy cannot send the unknown handles that `PassengerEnv` now
  warns on and ignores (C6.5).

## Findings

Code review findings, PR review comments and TODOs added in the PR, as of 24b61b7c (`main` 3a66efe9).

- Code review findings (40-46) were added on 2026-11-02, at 24b61b7c; their line numbers refer to it.
- PR review comments (1-12) were added on 2026-10-28. Their line numbers refer to the commit the comment was made on:
  1-4 to 14f2ab3e, 5-10 to fb6d6a52, 11-12 to an earlier version of fb6d6a52 (before a rebase).
- Items of the "Remaining" table in the PR description (13-15) were added on 2026-10-28.
- Gaps of the spike against the TravelWise [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md)
  (16-22, source "data model gap") and findings on the data model itself in flatland-book (23-33, source "data model
  review") were added on 2026-10-28. 23-33 belong in a flatland-book PR or issue, not in this repo.
- 34 and 35 (source "planning") were added on 2026-10-29 from decisions of this plan (phases 0 and 5).
- TODOs added in the PR (36-39) were added on 2026-11-02. Grouped with an existing finding: the corridor graph (34),
  planning connections with the query view (7). TODOs referring to other issues are tracked there: #527
  (`flatland/envs/rail_env_policies.py`), #553 (`flatland/envs/graph/graph_routing.py`).
- Findings keep the path and line they were reviewed at. Renames since: `flatland/envs/travelwise_overlay_env.py` is
  `flatland/envs/passenger_env.py` (c02af811), and since 052da2a6 `flatland/envs/travelwise/passenger_env.py`;
  `tests/test_travelwise_passenger_overlay_spike.py` is `tests/flatland/envs/travelwise/test_passenger_env_spike.py`;
  `rail_env_heading_info_wrapper.py` is gone (d1283828).
- "resolved" in Source marks threads resolved on GitHub. ❌ next to ✓ marks a finding fixed in the code whose PR
  comment thread is not resolved on GitHub yet.

| Fixed | ID | Source | Category | Severity | Finding | Commit message |
|---|---|---|---|---|---|---|
| ✓ | 1 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047200328), resolved | design | medium | `flatland/envs/travelwise_overlay_env.py:173`: make the env compliant with `AbstractRailEnv`. `step()` returned `(rail_env_dones, overlay_dones)`. Fixed in fb6d6a52: `step()` now returns `(obs, rewards, dones, info)` as per `Environment`. Accepting an `AbstractRailEnv` as rail env followed in f9e1f98d (finding 10). | refactor(envs): make TravelwiseOverlayEnv implement the Environment interface. |
| ✓ | 2 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047216530), resolved | design | medium | `flatland/envs/travelwise_overlay_env.py:29`: the class should inherit from `Environment` at least. Fixed in fb6d6a52: `TravelwiseOverlayEnv(Environment)` with `reset`/`step`/`get_agent_handles`/`clone_from`. | refactor(envs): make TravelwiseOverlayEnv implement the Environment interface. |
| ✓ | 3 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047229163), resolved | design | low | `flatland/envs/travelwise_overlay_env.py:123`: verify the terminology of the shadow `overlay_env` `RailEnv` against the [TravelWise data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md). Fixed in 03581a66: the shadow `overlay_env` is gone. The terminology check for the whole module is open in finding 5. | refactor(envs): drop TravelwiseOverlayEnv's shadow overlay_env RailEnv. |
| ✓ | 4 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047242428), resolved | design | medium | `flatland/envs/travelwise_overlay_env.py:60`: `rail_env` next to `overlay_env` is confusing; should `overlay_env` be `passenger_env`? Fixed in 03581a66: `overlay_env` is gone, only the underlying `rail_env` remains. The "overlay" naming is part of the terminology check in finding 5. | refactor(envs): drop TravelwiseOverlayEnv's shadow overlay_env RailEnv. |
| ✓ | 5 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047942249), resolved | design | medium | `flatland/envs/travelwise_overlay_env.py:27`: check terminology and modelling against the [TravelWise data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md) (overlay agent vs. passenger, overlay env, stops). Fixed in c02af811: the module is now `flatland/envs/passenger_env.py` with class `PassengerEnv`; overlay agents are passengers, the underlying env is the rail env with its vehicles, `overlay_stops` are `passenger_journeys`, `overlay_policy` is `passenger_policy`, and the test is `tests/test_passenger_env_spike.py`. The modelling points stay open in finding 8 (`PassengerConfiguration`); finding 10 (f9e1f98d) generalized the rail env to `AbstractRailEnv`. | refactor(envs): align TravelwiseOverlayEnv terminology with the TravelWise data model. |
| ✓ | 6 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047945105), resolved | design | medium | `flatland/envs/travelwise_overlay_env.py:160`: refine rewards, dones and infos. `step()` returns the underlying env's tuple verbatim; overlay agents get no rewards, dones or infos of their own. [Refined](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4403466565): implement the `Environment` interface's `step()` formally for passengers, but with empty rewards, dones and infos for now. Fixed in 56fccf5e: `get_agent_handles()` returns the passengers' handles; `step()` takes passenger actions (ignored for now), steps the rail env with a new `vehicle_policy` constructor argument, updates the passengers, and returns empty observations, rewards and infos with `dones == {'__all__': <rail env done>}`; `reset()` returns empty dicts. The vehicles' rewards and infos stay on the rail env. | feat(envs): implement PassengerEnv's step() interface for passengers with empty rewards, dones and infos. |
| ✓ | 9 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047964307), resolved | design | low | `flatland/envs/travelwise_overlay_env.py:16`: `OVERLAY_EARLIEST_DEPARTURE = 2` works around `RailEnv`'s departure edge cases at earliest departure 0 and 1. Bad smell: fix those edge cases where they come from. Fixed in 284418af: the constant is gone, with the schedule derived from it (`passenger_earliest_departures`/`passenger_latest_arrivals`/`passenger_earliest_departure`/`passenger_latest_arrival`, computed with grid-only column distances). The workaround was obsolete since 03581a66, as passengers are no longer `RailEnv` agents. Passengers now show up at their first stop on the first step; all later steps of the test traces are unchanged. The `RailEnv` 0/1 departure quirk is tracked in issue #280. Per-passenger departures as input (the itinerary in the data model) are not part of this fix; they are part of the passenger timetable in finding 13. | refactor(envs): drop PassengerEnv's PASSENGER_EARLIEST_DEPARTURE and its derived schedule. |
| ✓ | 10 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047981178), resolved | design | medium | `flatland/envs/travelwise_overlay_env.py:57`: `rail_env` should be typed `AbstractRailEnv`, to support graph envs. The env also reads grid internals (`rail_env._elapsed_steps`, `current_entry_point[0]` as a grid position). Fixed in f9e1f98d: `rail_env` is an `AbstractRailEnv`, journeys are entry points, and locations are compared as resources via `rail_env.resource_map`. `RailEnvHeadingInfoWrapper` returns entry points. A new test runs the scenario on a `GraphRailEnv` and gets the same trace as on the grid. `rail_env._elapsed_steps` stays: it is an `AbstractRailEnv` attribute, so it works on graphs. `PassengerEnv` still wraps the rail env instead of extending it, as the data model has it. | refactor(envs): accept any AbstractRailEnv as PassengerEnv's rail env. |
| ✓ | 11 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4098370551), resolved | docs | low | `flatland/envs/travelwise_overlay_env.py:30`: docstring wording suggestion ("the underlying env's rail topology, but are otherwise independent of it - separate agents,"). Fixed in b9c6d084: suggestion applied. | chore: reduce mental overload. |
| ✓ | 12 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4098588944), resolved | correctness | medium | `flatland/envs/rail_env_heading_info_wrapper.py:45`: `heading` holds only the train's final waypoint. A passenger going A -> B on a train A -> B -> C never boards (`travelwise_overlay_env.py:215` compares against the next stop). Intermediate stops are neglected for now. The reply asks to make this and further assumptions explicit and to compile a prioritized list: grid not graph, underlying trains have no alternatives at intermediate stops, A/B/C are station tracks, not stations. Grid not graph is addressed by f9e1f98d (finding 10). [Refined](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4403558487): add a passenger timetable (source, target, earliest departure, latest arrival), a passenger policy respecting the earliest departure, and passenger rewards (as in finding 13). Fixed in 14578ec9, d1283828, 2711d24d, 93a6de3f and d2837293: the heading wrapper is gone (d1283828), vehicles announce their remaining stops (`TimetablePredictor`), passengers have journeys with earliest departure and latest arrival (14578ec9), a passenger policy boards at intermediate stops and respects the earliest departure (2711d24d), passengers get rewards (93a6de3f) and change trains (d2837293). The assumptions are explicit in `PassengerEnv`'s docstring (Limitations) and in this plan (Limitations, Assumptions): graph only, station tracks not stations, the missed stop at another alternative. | feat(envs): let passengers hop on and off via actions, decided by a passenger policy on their observations. |
| ✓ | 13 | PR description (Remaining) | design | low | `flatland/envs/travelwise_overlay_env.py:57`: `overlay_policy` is a required constructor argument, but its actions are ignored (docstring at line 38). Still open after the rename in c02af811: `passenger_policy` in `flatland/envs/passenger_env.py:52`. Refined in the PR description's "Remaining" table (no review thread): add a passenger timetable (source, target, earliest departure, latest arrival), a passenger policy respecting the earliest departure, and passenger rewards. Fixed in 14578ec9, 2711d24d and 93a6de3f: journeys with earliest departure and latest arrival (14578ec9); the unused `passenger_policy` argument is gone, a passenger policy outside the env decides the passengers' actions, the earliest departure is its decision (2711d24d); rewards for lateness and early boarding (93a6de3f). | feat(envs): let passengers hop on and off via actions, decided by a passenger policy on their observations. |
| ✓ | 14 | PR description (Remaining) | test-coverage | medium | `tests/test_passenger_env_spike.py:247`: the test runs a single overlay agent (one A -> B -> C route). No test covers several overlay agents sharing the same trains. Fixed in 721ad3f2: `test_multiple_passengers_share_trains` (`tests/test_passenger_env_spike.py:424`) runs five passengers on the four trains: two board one train together, three wait at B and only the one heading to A boards leg 2. It asserts each passenger's mode changes and arrivals. It runs on the grid only; the graph test keeps one passenger. | test(envs): cover multiple concurrent passengers in PassengerEnv. |
| ✓ | 17 | data model gap | design | low | `flatland/envs/passenger_env.py:52`: the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md) has passenger journeys as a `Line` with `LineFlexibleStoppingPoint`s (alternative stopping points per stop). `passenger_journeys` holds one fixed entry point per stop. Fixed in 14578ec9: `PassengerJourney.waypoints` are groups of alternative entry points, as `EnvAgent.waypoints` (the data model's `LineFlexibleStoppingPoint`). | feat(envs): give passengers single-ride journeys with earliest departure and latest arrival. |
| ✓ | 35 | planning (the plan, phase 5) | design | medium | `flatland/envs/passenger_env.py`: single ride only (decided for phase 2, Q2 in the plan). A journey with a change of trains (A -> B on one train, B -> C on another, as passenger 0 of the shuttle tests) never arrives. Planned in phase 5: intermediate waypoint groups in `PassengerJourney.waypoints` (the vehicles' shape) for the stops where the passenger changes trains; the boarding rule (a vehicle's future stop group shares an entry point with the passenger's next stop, Q10) applies per ride. Fixed in d2837293: passenger 0 of the shuttle tests goes A -> B -> C and arrives. | feat(envs): support passenger journeys with changes of trains at given stops. |
|  | 7 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047955787) | design | medium | `flatland/envs/travelwise_overlay_env.py:136`: refine passenger actions/policy while the agent waits (mode 2). Today it boards the first train heading to its next stop, without a decision of its own. Related to finding 13. [Refined](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4403466997): add a query view and a passenger policy following it, including hopping on at intermediate stops (see finding 12). Partly addressed in 2711d24d: `BoardForTargetPassengerPolicy` decides `HOP_ON`/`HOP_OFF`/`WAIT` on the passenger's observation, including hopping on and off at intermediate stops; since d2837293 also at changes. Still open: the query view (all vehicles' future stops) - the observation holds per-vehicle predictions only. Same concern: the TODO in `flatland/envs/travelwise/passenger_env.py:119` (plan connections with the query view). | feat(envs): add a query view and a passenger policy following it to PassengerEnv. |
|  | 8 | [PR comment](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4047962172) | design | low | `flatland/envs/travelwise_overlay_env.py:24`: use `GraphRailEnv` as the rail env once #542 (edge lengths) is merged. With it, model the `OverlayMode` tuple as a `PassengerConfiguration`; modes 1/2/3 can be derived from it. Since f9e1f98d the tuple is `PassengerMode` and holds the rail env's resource as the passenger's location. A resource is the trains' occupancy unit, not a passenger location: it can be coarser or finer than a stopping point. [Refined](https://github.com/flatland-association/flatland-rl/pull/528#discussion_r4403462760): hold the configuration (entry point) instead of the resource in `PassengerConfiguration`. Partly addressed in f90b72da: the spike and its tests run on `GraphRailEnv` only. The `PassengerConfiguration` holding the entry point is still open (`PassengerMode` still holds the resource), waits for #542 (open). | refactor(envs): use GraphRailEnv and a PassengerConfiguration holding the entry point in PassengerEnv. |
|  | 15 | PR description (Remaining) | style | low | `tests/test_travelwise_passenger_overlay_spike.py`: the test module is still named a "spike" (`tests/test_passenger_env_spike.py` since c02af811). Promote it (rename, drop the exploratory framing) once the design is stable. Open: still `tests/flatland/envs/travelwise/test_passenger_env_spike.py`. | refactor(envs): promote PassengerEnv spike out of spike status. |
|  | 16 | data model gap | design | medium | `flatland/envs/passenger_env.py:103`: the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md) has a `TravelWisePolicyRunner` that steps the rail env with a Flatland policy, then the passenger env with a passenger policy based on the new rail state. `PassengerEnv.step()` steps the rail env itself and updates passengers as a side effect; there is no separate passenger step and no runner. Related to finding 6. Partly addressed by 56fccf5e: `PassengerEnv.step()` now steps the rail env with its `vehicle_policy`, then updates the passengers - the runner lives inside the env. Since 2711d24d the passenger policy runs outside the env, in the controller (for now the test loop), which steps `PassengerEnv` with the passengers' actions; `PassengerEnv` steps the vehicles with their policy by design (the plan's controller and step). Still open: the runner as a library class, planned in C6.1. | feat(envs): add a TravelWise policy runner stepping the rail env and then the passenger env. |
|  | 18 | data model gap | design | low | `flatland/envs/passenger_env.py:20`: the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md) has `PassengerEnv` inherit from `RailEnv` (travelwise.md: from `InfrastructureEnv`) and refer to a rail env. `PassengerEnv` uses composition only. Either inherit, or drop the inheritance from the data model. Open: composition is kept by design; the plan states the divergence (Terms), the data model is unchanged. | docs(travelwise): model PassengerEnv as composition over the rail env. |
|  | 19 | data model gap | design | medium | `flatland/envs/passenger_env.py:63-67`: the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md)'s `EnvAgent` has `current()`, `next_configuration()`, `timetable` and `next_stop()`. Passengers are not `EnvAgent`s: their state is in separate per-passenger dicts (`passenger_current_entry_point`, ...), and the next stop is only the internal `_passenger_stop_index`. Partly addressed in 2711d24d: `PassengerEnv.next_stop()` is public. Still open: the per-passenger dicts. | refactor(envs): give passengers an EnvAgent-like API with current/next configuration and next stop. |
|  | 20 | data model gap | design | medium | `flatland/envs/passenger_env.py:20`: the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md)'s `RailEnv` has `effects_generator`, `events_generator` and `register()`/`deregister()` for event topics, which `PassengerEnv` would inherit. The spike has none of them on the passenger side, so passenger milestones and events can't be raised. Open. | feat(envs): raise passenger events in PassengerEnv. |
|  | 21 | data model gap | design | low | `flatland/envs/passenger_env.py:75`: in the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md), `reset()` re-generates rail, line and timetable. `PassengerEnv.reset()` resets the rail env only; passenger journeys are fixed constructor input, with no passenger line or timetable generator. Open. | feat(envs): add a passenger journey generator to PassengerEnv. |
|  | 22 | data model gap | design | low | `flatland/envs/passenger_env.py:20`: the [data model](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise/data_model.md) collects the run in a trajectory. `PassengerEnv` is not integrated with trajectories or `PolicyRunner`. Open. | feat(trajectories): record PassengerEnv runs in trajectories. |
|  | 23 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: Journey and Itinerary don't match [travelwise.md](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise.md). There a journey is origin and destination, and an itinerary a sequence of modes of transport (legs) that can be updated. The data model equates Journey with `Line` (stops served by a vehicle) and Itinerary with `Timetable` (time windows). `Leg` and `Transfer` (a defined term in travelwise.md) are missing. Open. | docs(travelwise): add Journey, Itinerary, Leg and Transfer to the data model. |
|  | 24 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: the passenger state is missing: hop on/hop off (travelwise.md), a passenger configuration (a stopping point while waiting, a vehicle while riding), the passenger graph (a subgraph of the infrastructure graph) and the group size (N=1..n). The spike needed three modes: off map, waiting, riding. Open. | docs(travelwise): add the passenger configuration and state to the data model. |
|  | 25 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: passenger actions are a TODO ("what are the actions"), though travelwise.md answers it (move to edge; hopping on is implicit). Passenger observations (global view, list of itineraries), rewards/metrics (shortest time, fewest transfers, longer transfer times) and the passenger timetable (source, target, earliest departure, latest arrival) are missing. Open. | docs(travelwise): add passenger actions, observations, rewards and timetable to the data model. |
|  | 26 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: [travelwise.md](https://github.com/flatland-association/flatland-book/blob/main/resources/travelwise.md) names the generalized env `InfrastructureEnv`, has `PassengerEnv` inherit from it, and calls both together `TravelWiseEnv`. The data model only has `RailEnv`, has `PassengerEnv` inherit from it, and has no composite env. Open. | docs(travelwise): align env names and inheritance in the data model with travelwise.md. |
|  | 27 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: multimodality is missing: travelwise.md has modes of transport (train, bus, plane, walkway) on edges, conditional paths (walkways for reduced mobility, fast-track lanes) and optional capacities. Neither `TransitionMap` nor `EnvAgent` has a mode or vehicle type. Open. | docs(travelwise): add modes of transport, conditional paths and capacities to the data model. |
|  | 28 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: milestones, events and messages to the external Travel Wise layer, the core output in travelwise.md, are missing. The diagram has only `register()`/`deregister()` for event topics. Open. | docs(travelwise): add milestones and events to the data model. |
|  | 29 | data model review | docs | low | flatland-book `resources/travelwise/data_model.md`: the demand generator that travelwise.md envisages (generalizing line and timetable generators to journeys and itineraries) is missing. Open. | docs(travelwise): add a demand generator to the data model. |
|  | 30 | data model review | docs | low | flatland-book `resources/travelwise/data_model.md`: `ResourceMap`, malfunctions, `ObservationBuilder`, `Rewards` and `Trajectory` are mentioned in the text or used by Flatland, but missing from the diagram. Open. | docs(travelwise): add resource map, malfunctions, observations, rewards and trajectory to the data model diagram. |
|  | 31 | data model review | docs | medium | flatland-book `resources/travelwise/data_model.md`: a waiting passenger needs to know which stops a vehicle still serves (the spike's `RailEnvHeadingInfoWrapper`, finding 7's query view, finding 12's intermediate stops). The diagram reaches this only through `EnvAgent.timetable`; a query view is missing. Open. | docs(travelwise): add a vehicle query view to the data model. |
|  | 32 | data model review | docs | low | flatland-book `resources/travelwise/data_model.md`: the data model still says Configuration, though flatland-rl renamed it to EntryPoint (7599b962). `TransitionMap.configurations: Set[Coordinate]` mixes a cell coordinate with a configuration (which includes the direction). `EnvAgent.current()`/`next_configuration()`/`offset` don't match the code's `current_entry_point`/`next_entry_point`/speed counter. Open. | docs(travelwise): align data model terms with flatland-rl's entry points. |
|  | 33 | data model review | docs | low | flatland-book `resources/travelwise/data_model.md`: diagram inconsistencies: `LineFlexibleStop` (in `AgentTimetable.timetable`, `EnvAgent.next_stop()`) vs. the class `LineFlexibleStoppingPoint`; `AgentTimeTable` vs. `AgentTimetable`; the typo `latesttArrival`; `Timetable` relates to `AgentTimetable` only through an attribute, with no edge. Open. | docs(travelwise): fix naming inconsistencies in the data model diagram. |
|  | 34 | planning (the plan, phase 0) | test-coverage | low | `tests/test_passenger_env_spike.py`: the corridor graph is built from a `GridTransitionMap` with `GraphTransitionMap.grid_to_digraph()` (decided for phase 0): no `RailEnv`, but a grid as construction aid. Build the graph natively (nodes, edge `actions`, node `prohibited_actions`) and its resource map without a grid. Open: the TODO is in `tests/flatland/envs/travelwise/test_passenger_env_spike.py:85` (f90b72da). | test(envs): build the passenger env spike's corridor graph without a grid. |
|  | 36 | TODO | design | low | `flatland/envs/rail_env_policies.py:52`, `flatland/envs/travelwise/passenger_env_observations.py:17`: the elapsed steps are in the observations of the vehicles and the passengers, but belong into the infos (Q18). Waits for the rail env to hand them out (Q20). Open. | refactor(envs): move the elapsed steps from the observations to the infos. |
|  | 37 | TODO | design | low | `flatland/envs/travelwise/passenger_env.py:83`: passenger rewards can be positive (an early arrival), unlike `DefaultRewards`. Align with them or keep the bonus once the TravelWise rewards are settled. Open. | fix(envs): align the passenger rewards with Flatland's DefaultRewards. |
|  | 38 | TODO | correctness | medium | `flatland/envs/travelwise/passenger_env.py:121-122`: a passenger may board a vehicle where it doesn't stop (passing through), and hop off at any step. Board and alight only where the vehicle stops. Open. | feat(envs): let passengers board and alight only where the vehicle stops. |
|  | 39 | TODO | test-coverage | low | `flatland/envs/travelwise/passenger_env.py:125`: no test covers a vehicle stopping at another alternative of the passenger's next stop group (the passenger stays on board), as every group on the test corridor has one entry point. Open. | test(envs): cover a passenger missing its stop at another alternative of a group. |
|  | 40 | code review | bug | medium | `flatland/envs/rail_env_policies.py:119,147`: `ShortestPathPolicy` routes each agent once and keeps the route for good. Reused for a second episode (another line), it follows the first episode's routes and fails with "off its route". On `main` the policy failed as well (`IndexError`), but only for agents not done in the first episode, as it dropped the routes of agents once done. Reset the routes when a new episode starts (e.g. `elapsed_steps == 0`), and cover a policy reused across two episodes in a test. Open: planned in C6.2. | fix(policies): reset ShortestPathPolicy's routes when a new episode starts. |
|  | 41 | code review | bug | medium | `flatland/envs/travelwise/passenger_env_policies.py:48`: `_may_depart()` finds the group the passenger waits at by `journey.waypoints.index(next_stop) - 1`, the first match. On a journey visiting a group twice (A -> B -> A), waiting at B it gets index -1, the target's `None`, and boards before its earliest departure at B. `PassengerEnv` itself counts the stop index correctly. Hand the earliest departure (or the stop index) out in the observation instead. Repeated visits are tracked in [#557](https://github.com/flatland-association/flatland-rl/issues/557). Open: planned in C6.4. | fix(envs): let the passenger policy read the earliest departure of the group it waits at from its observation. |
|  | 42 | code review | robustness | low | `flatland/envs/travelwise/passenger_env.py:233,262,266`: the docstring says invalid actions are ignored. A `HOP_ON` with a negative vehicle handle indexes from the end (`-1` boards the last vehicle, if at the passenger's resource); a handle out of range raises `IndexError`; an action that is not an `(action, handle)` tuple fails to unpack. Check the handle against `rail_env.get_agent_handles()` and ignore malformed actions. Open: planned in C6.5. | fix(envs): ignore passenger actions with unknown vehicle handles. |
|  | 43 | code review | robustness | medium | `flatland/envs/rail_env_policies.py:154`: with an unreachable waypoint group, the router returns a truncated route, and `ShortestPathPolicy` follows it silently: it moves forward past its end, and the last routed intermediate stop counts as the target (`_stop_indices[1:-1]`), so it is not stopped at. On `main` the policy asserted that a path was found. Fail (or warn) when fewer groups are routed than given, and cover it in a test. Open: planned in C6.3. | fix(policies): fail when ShortestPathPolicy cannot route through all waypoint groups. |
|  | 44 | code review | type-hints | low | `flatland/envs/travelwise/passenger_env_observations.py:29`: `PassengerObservation.mode` spells out `PassengerMode` again. `flatland/envs/travelwise/passenger_env_policies.py:31`: `act(journey: PassengerJourney = None)` is not `Optional`, and `_may_depart()` fails on `None` - make `journey` required. `flatland/envs/travelwise/passenger_env.py:153`: `vehicle_observations: Dict[int, object]` can be `Dict[int, VehicleObservation]`. Open: planned in C6.7. | refactor(envs): tighten the type hints of the passenger env. |
|  | 45 | code review | performance | low | `flatland/envs/travelwise/passenger_env_observations.py:57-58`: per passenger and vehicle, `get()` computes `future_stops_serving()` twice: once in `serves()`, once for the value. Compute it once and keep non-empty results. Open: planned in C6.8. | perf(envs): compute a vehicle's stops serving a passenger once per observation. |
|  | 46 | code review | style | low | Two long methods: `ShortestPathPolicy._act()` (`flatland/envs/rail_env_policies.py:167`) mixes stopping at intermediate stops with steering; extract the stop decision into its own method. `UnilateralTimetablePredictor._predict()` (`flatland/envs/travelwise/timetable_predictor.py:109`) mixes the first step (off map, halted, moving) with the loop over the route; extract the first step. The greedy loops of the grid and graph routers duplicate each other, tracked in [#553](https://github.com/flatland-association/flatland-rl/issues/553). Open: planned in C6.6. | refactor(policies): extract the stop decision and the first projected step into their own methods. |

Issue references: the PR description links no issue, so the PR closes none. The `main..HEAD` diff references #527
and #553 (TODOs, see above); the PR closes neither.

## Decisions

- ✅ Q1 Routing: a shared greedy `networkx` router over `rail.g` (nearest alternative per group), injected into
  `ShortestPathPolicy` and the predictor, so both pick the same entry points.
- ✅ Q2 Existing shuttle tests keep their setup; passenger 0 (A -> C) never arrives with single rides, and becomes
  A -> B -> C in phase 5.
- ✅ Q3 Boarding only from the passenger's earliest departure on - since Q23 the passenger policy's decision, not
  checked by the env.
- ✅ Q5 Layout: `flatland/envs/travelwise/` with `passenger_env.py`, `passenger_env_action.py`,
  `passenger_env_observations.py`, `passenger_env_policies.py`, `timetable_predictor.py`, following
  `rail_env.py`/`rail_env_action.py`/`rail_env_policies.py`; no `PassengerEnvPolicy` base class.
- ✅ Q6 An arrival in the step the rail env finishes is rewarded as an arrival.
- ✅ Q7 The test timetable generator takes a departure time per stop.
- ✅ Q8 The late passenger is the A -> B passenger on T0.
- ✅ Q9 The passenger without a vehicle is C -> A.
- ✅ Q10 Boarding rule: a vehicle's future stop group shares an entry point with the passenger's next stop.
- ✅ Q11 Observations are empty after `reset()` and hold only dynamic data; static data via the infos (Q19).
- ✅ Q12 Data shared by all agents (positions and plans) is one immutable object referenced by every observation: with
  n vehicles, a copy per vehicle would be n * n plans.
- ✅ Q13 The plan is the projection, per entry point with stop marks; the timetable is static data.
- ✅ Q14 The plans are the env's prediction (same router as the vehicle policy), not the policy's intent.
- ✅ Q15 Passengers are off the map after `reset()`.
- ✅ Q16 The vehicles' dedicated observation comes first, as C3.0.
- ✅ Q17 `VehicleObservationBuilder` lives in `flatland/envs/travelwise/` for now.
- ✅ Q18 The vehicle's state (`done`, entry points, `is_cell_exit`) is dynamic: it goes into the observation. The
  elapsed steps belong into the infos, but the rail env's infos do not have them, and adding them is a rail env
  change: for now they go into the observation too (`ShortestPathObservation.elapsed_steps`). TODO: move them to the
  infos with the rail env change of Q20.
- ✅ Q19 Static data reaches the policies through the infos, as a reference to one immutable object; runners pass the
  infos to `act_many()` through `**kwargs`. Follows RLlib (infos keyed by agent id, `"__common__"` for data shared
  by all agents; the old API's `Policy.compute_actions(..., info_batch=...)`) and Flatland's own decision data in the
  infos (`action_required`); gymnasium describes infos as auxiliary.
- ✅ Q20 Who puts the vehicles' static data into the infos: Flatland's infos are `infos[key][handle]` (RLlib:
  `infos[handle][key]`), and `PolicyRunner` persists `{key: infos[key][handle]}` per agent and step into the
  trajectory - static data in the rail env's infos would be persisted every step, and the map does not serialize.
  Decided: the rail env's infos stay unchanged in this PR. `PassengerEnv`, the vehicles' runner, adds the vehicles'
  static data to the infos it passes to the vehicle policy (`infos["static"]`, the role of an RLlib env-to-module
  connector), and hands out the passengers' static data in its own infos. Other runners (`PolicyRunner`) keep
  using `ShortestPathPolicy` with `FullEnvObservation` until the rail env hands out its static data (TODO, a core
  change with trajectory persistence skipping it).
- ✅ Q22 Backwards compatibility of `ShortestPathPolicy` with `FullEnvObservation` (the env as observation): one
  helper - if an observation is the env, `act_many()` first derives the `ShortestPathObservation` and the static data
  from it, then runs the same code as for the new observation; one code path for the routing and stop logic.
- ✅ Q23 Boarding before the earliest departure: the env does not enforce it (the env is dull, the policy decides); the
  rewards penalize it per step boarded too early, `earliest departure - boarding step`, on the same scale as the
  lateness reward, so a policy gains nothing by ignoring the constraint.
- ✅ Q24 The runner is `TravelWisePolicyRunner`, the data model's name, in `flatland/envs/travelwise/`, for the
  passenger env only; a generic runner for any `Environment` waits for a second env needing it (decided 2026-11-03).
- ✅ Q25 The runner does not step the vehicles: `PassengerEnv` keeps stepping them with the vehicle policy (Controller
  and step), unlike the data model's rail env step before the passenger env step (decided 2026-11-03).
