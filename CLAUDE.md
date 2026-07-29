# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Flatland is a multi-agent grid/graph rail-network simulator (`RailEnv`) used for Multi-Agent Reinforcement
Learning research, plus tooling around it: rail/line/timetable generation, observations/predictions for RL
policies, rendering, trajectory recording/replay, and evaluation services for the AIcrowd/Flatland challenges.

## Commands

All commands assume dependencies from `requirements-dev.txt` (+ `requirements-ml.txt` for `flatland/ml`/`tests/ml`)
are installed, and are run from the repo root. `tox` wraps most of these into reproducible environments (see
`tox.ini`) — CI (`.github/workflows/checks.yml`) invokes tox directly, e.g. `tox -e py3.12,py3.12-verify-install`.

- **Run the core test suite** (matches CI's `test` job): needs `tests/regression/test_episodes_deadlock_avoidance.py`'s
  fixtures — a `flatland-baselines` checkout on `PYTHONPATH` and a `BENCHMARK_EPISODES_FOLDER` populated from the
  `FLATLAND_BENCHMARK_EPISODES_FOLDER` archive (see `flatland-benchmarks-episodes-url` in `checks.yml`):
  ```
  python -m pytest --ignore=tests/ml -m "not slow"
  ```
  Without those fixtures set up, drop the regression test or just run a narrower path, e.g.
  `python -m pytest tests/envs/test_foo.py`.
- **Run a single test**: `python -m pytest tests/path/to/test_file.py::test_name`.
- **Run the ML test suite** (`flatland/ml`, RL training — flaky, matches CI's `testml` job): needs `--retries`
  since training runs are inherently non-deterministic:
  ```
  python -m pytest tests/ml --retries 2 --retry-delay 5
  ```
- **Lint**: `flake8 flatland tests examples benchmarks` (config in `tox.ini`'s `[flake8]` section: max line length
  120, `docs` excluded, a fixed ignore list for whitespace/formatting codes). The CI `lint` job is currently
  disabled (`if: false`) but the config is still the source of truth for style.
- **Regenerate `requirements*.txt`** after changing `pyproject.toml` dependencies: `tox -e requirements`.
- **Check for dependency drift** (unused/missing/misdeclared deps across the `flatland`/`flatland/ml`/`tests`
  boundary): `tox -e py3.13-verify-requirements` (uses `deptry`).
- **Notebooks** (in `notebooks/`, executed as smoke tests): `tox -e py3.12-notebooks`.
- **Full tox matrix**: `tox` (runs everything across Python 3.10–3.13 — slow; prefer targeted `pytest`/`flake8`
  invocations above during iteration).

## Architecture

### Grid and graph are parallel implementations of one rail-network abstraction

`flatland/core/` defines the rail network purely in terms of generic abstractions, each with two independent
concrete implementations — grid-based (`flatland/envs/grid/`) and graph-based (`flatland/envs/graph/`). Neither
is a layer on top of the other; they're alternative representations selected at env-construction time. A
**configuration** is each representation's network-position primitive: `((row, col), direction)` for grid,
a node-id string for graph.

- **`TransitionMap`** (`core/transition_map.py`) — the rail topology: `get_transitions`/`set_transitions`,
  `get_successor_configurations`/`get_predecessor_configurations`/`is_valid_configuration`. Grid impl:
  `GridTransitionMap` → `RailGridTransitionMap`. Graph impl: `GraphTransitionMap`.
- **`ResourceMap`** (`core/resource_map.py`) — maps a configuration to the "resource" (occupancy unit) used for
  conflict detection. Grid's `GridResourceMap` normally resolves to `(row, col)`, but for level-free crossings
  returns `(position, direction % 2)` so the two crossing axes count as distinct resources. Graph's
  `GraphResourceMap` is identity on the node id.
- **Distance-map subsystem** (`core/configuration_distance_map.py`, `core/distance_map.py`,
  `core/distance_map_walker.py`) — `ConfigurationDistanceMap` (agent-agnostic, keyed by
  `(source_config, target_config)`) → `AgentSourceTargetDistanceMap` (agent-handle-aware, shared template-method
  `_compute()`) → concrete `DistanceMap` (grid, numpy array) / `GraphDistanceMap` (graph, nested dict).
  `DistanceMapWalker` does the backward BFS that fills it in, one independent walk per target configuration
  (a shared visited set across targets silently breaks on cyclic/looped layouts). `RailEnv` keeps one
  `distance_map` instance alive across `reset()` calls rather than reconstructing it per episode.

### `flatland/envs/` — env control flow

`RailEnv`/`AbstractRailEnv` (`rail_env.py`) is the env facade, generic over `(TransitionMap, ResourceMap,
ConfigurationType)`. `reset()` calls `rail_generators.py` (topology), `line_generators.py` (agent start/target
assignment), `timetable_generators.py` (departure/arrival windows). Per `step()`, for each agent: derive the
desired next configuration from the action via the `step_utils` state machine (`TrainState`/
`TrainStateMachine`); look up both agents' current/next *resources* via `resource_map.get_resource(...)`; feed
`(current_resource, new_resource)` pairs into `agent_chains.py`'s `MotionCheck`, which resolves cross-agent
conflicts (head-on swaps, same-target collisions) once all agents for the step are registered; then finalize
state/position/rewards. `EnvAgent` (`agent_utils.py`) holds per-agent state; `observations.py`/`predictions.py`
build the observation returned to policies, typically via the distance map's shortest paths.

### `flatland/ml/`

A thin adapter layer over `RailEnv` for RL frameworks, not a parallel env implementation: `wrapped_rail_env.py`
wraps a `RailEnv`; `ml/pettingzoo/` and `ml/ray/` adapt it to PettingZoo's `ParallelEnv` and RLlib's
`MultiAgentEnv` respectively; `ml/observations/` adapts Flatland's tree observation to fixed-shape gym spaces.

### Persistence

`RailEnvPersister` (`envs/persistence.py`) pickle/msgpack-serializes full `RailEnv` state (rail, agents,
optionally distance maps) to a single file — one env snapshot. `trajectories/trajectories.py`'s `Trajectory` is
a higher-level, tabular (pandas) episode recorder: per-step actions/positions/rewards/dones as data frames,
plus `RailEnvPersister`-saved env snapshots so a trajectory can be replayed/resumed from any recorded step, not
just step 0.

### Other top-level dirs

- `callbacks/` — episode-lifecycle hooks (e.g. movie generation).
- `evaluators/` — AIcrowd competition evaluation service/client + trajectory-based evaluators.
- `integrations/interactiveai/` — REST API client for the InteractiveAI dashboard.
- `utils/` — rendering (`rendertools.py`, `editor.py`), grid helpers, seeding.
- `env_generation/` — higher-level convenience env-builder over the generators.
- `png/`, `svg/` — static rendering image assets, not code.

## Conventions (see `CONTRIBUTING.md` for full detail)

- Prefer `NamedTuple` over a plain unnamed `Tuple` or `Dict` for structured data that doesn't need methods.
- Use `attrs` (`@attrs`/`attrib`) for classes that must keep multiple members in sync as an invariant.
- Use `abc.ABCMeta`/`abc.abstractmethod` for extension-point base classes.
- Docstrings follow numpydoc format; type hints are expected throughout (PEP 484).
- Avoid currying/closures to encapsulate state — prefer a class when the object needs multiple methods.
