# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Flatland is a multi-agent grid/graph rail-network simulator (`RailEnv`) used for Multi-Agent Reinforcement
Learning research, plus tooling around it: rail/line/timetable generation, observations/predictions for RL
policies, rendering, trajectory recording/replay, and evaluation services for the AIcrowd/Flatland challenges.

## Commands

All commands assume dependencies from `requirements-dev.txt` (+ `requirements-ml.txt` for `flatland/ml`/`tests/ml`)
are installed, and are run from the repo root. `tox` wraps most of these into reproducible environments (see
`tox.ini`) — CI (`.github/workflows/checks.yml`)'s containerized jobs run tox's own testenvs directly via
`tox run -e <env> --current-env` (the `tox-current-env` plugin: skips venv creation/dependency reinstall against
an image's already-installed deps, but still applies the testenv's `set_env`/`commands` verbatim), so `tox.ini`
is the single source of truth for both local and CI runs — editing it changes CI behavior without touching
`checks.yml`. Two gotchas this implies: (1) a `[testenv:...]` section that declares its own `set_env` *replaces*
rather than merges with the base `[testenv]` section's `set_env` (tox does not merge these) — any testenv needing
extra env vars must start its `set_env` with `{[testenv]set_env}` or it silently loses `PYTHONPATH`/
`CMAKE_POLICY_VERSION_MINIMUM`; (2) a dev dependency that's only invoked via CLI and never `import`ed (e.g. `tox`,
`tox-current-env`) must be added to the `verify-requirements` testenv's `DEV_MODULES` list or `deptry` (`DEP002`)
flags it as unused.

- **Run the core test suite** (matches CI's `test` job): needs `tests/test_flatland_regression_episodes.py`'s and
  `tests/test_flatland_evaluators_trajectory_analysis.py`'s fixture — a `flatland-baselines` checkout on
  `PYTHONPATH` and a `BENCHMARK_EPISODES_FOLDER` populated from the
  `FLATLAND_BENCHMARK_EPISODES_FOLDER` archive (see `flatland-benchmarks-episodes-url` in `checks.yml`, also
  duplicated in `benchmarks/benchmark_episodes.py`'s `DOWNLOAD_INSTRUCTIONS` — keep both in sync if the URL ever
  bumps to a new version). Download and point the env var at the extracted folder (any location works, it's
  read purely via `BENCHMARK_EPISODES_FOLDER`, nothing hardcodes a path):
  ```bash
  curl -sSL -o FLATLAND_BENCHMARK_EPISODES_FOLDER.zip "https://github.com/flatland-association/flatland-scenarios/raw/refs/heads/178-agents-living-on-the-edge-9/trajectories/FLATLAND_BENCHMARK_EPISODES_FOLDER_v7.zip"
  unzip -o -q FLATLAND_BENCHMARK_EPISODES_FOLDER.zip -d /path/to/FLATLAND_BENCHMARK_EPISODES_FOLDER
  export BENCHMARK_EPISODES_FOLDER=/path/to/FLATLAND_BENCHMARK_EPISODES_FOLDER
  python -m pytest --ignore=tests/ml -m "not slow"
  ```
  Without that fixture set up, the affected tests fail outright with an `AssertionError` naming
  `DOWNLOAD_INSTRUCTIONS` (not a skip) — so a run without it set looks like real regressions. Either set up the
  fixture, or drop the affected tests / run a narrower path, e.g. `python -m pytest tests/envs/test_foo.py`. If
  the `flatland-baselines` checkout lives inside this repo's own tree (as CI's `test` job does it, and as needed
  to put it on `PYTHONPATH` without an absolute path), add `--ignore=flatland-baselines` — otherwise pytest's
  default recursive collection picks up its test suite too.
- **Run a single test**: `python -m pytest tests/path/to/test_file.py::test_name`.
- **Run the ML test suite** (`flatland/ml`, RL training — flaky, matches CI's `testml` job): needs `--retries`
  since training runs are inherently non-deterministic:
  ```
  python -m pytest tests/ml --retries 2 --retry-delay 5
  ```
- **Lint**: `flake8 flatland tests examples benchmarks` (config in `tox.ini`'s `[flake8]` section: max line length
  120, `docs` excluded, a fixed ignore list for whitespace/formatting codes). The CI `lint` job is gated on the
  `LINT_ENABLED` repo/org Actions variable (`.github/workflows/checks.yml`'s `if: ${{ vars.LINT_ENABLED ==
  'true' }}`) — unset means disabled — but the config is still the source of truth for style.
- **Regenerate `requirements*.txt`** after changing `pyproject.toml` dependencies: `tox -e requirements`.
- **Check for dependency drift** (unused/missing/misdeclared deps across the `flatland`/`flatland/ml`/`tests`
  boundary): `tox -e py3.13-verify-requirements` (uses `deptry`).
- **Notebooks** (in `notebooks/`, executed as smoke tests): `tox -e py3.12-notebooks`.
- **Full tox matrix**: `tox` (runs everything across Python 3.10–3.14 — slow; prefer targeted `pytest`/`flake8`
  invocations above during iteration).
- **Verify the Cython build actually compiles** (see "Cython-accelerated hot paths" below):
  `tox -e py3.12-verify-cython-build`. To manually build in place and check (Cython auto-provisions via
  `[build-system] requires` - no need to `pip install cython` yourself first): `python -c "from setuptools
  import setup; setup()" build_ext --inplace`, then confirm e.g. `python -c "import
  flatland.envs.step_utils.state_machine as m; assert m.__file__.endswith('.so')"`. Clean up compiled artifacts
  afterward with `rm -rf build && find flatland \( -name '*.c' -o -name '*.so' \) -print0 | xargs -0 rm -f`
  (find-based rather than a hardcoded per-module glob, so it still catches every compiled artifact if an
  `ext-modules` entry moves out of `step_utils` or a new one is added elsewhere under `flatland/`). To force the
  plain-Python fallback instead (e.g. to reproduce
  `verify-build-no-gcc`'s behavior), fake a missing compiler: `CC=/nonexistent-cc CXX=/nonexistent-cxx python -c
  "from setuptools import setup; setup()" build_ext --inplace`.
- **Run tests against the compiled Cython build** (catches a `.pxd`/`.py` mismatch that compiles cleanly but
  breaks at runtime, e.g. a `cdef class` field only ever assigned in the `.py` source): build in place per
  above, then `python -m pytest --ignore=tests/ml -m cython_ext` - a fast, targeted subset (not the full suite)
  of tests marked `pytest.mark.cython_ext` for exercising `state_machine.py`/`states.py`/
  `rail_env_shortest_paths.py`. Wired into `tox -e py3.13-verify-cython-build-no-isolation`. Mark any new test
  that exercises these modules the same way - see `CONTRIBUTING.md`'s Cython section.
- **Profiling notebooks** (`benchmarks/flatland_performance_profiling.ipynb`,
  `benchmarks/benchmark_k_shortest_paths_profiling.ipynb`): `tox -e py3.13-profiling` /
  `tox -e py3.13-profiling-get-k-shortest-paths` — use Python 3.13, not 3.12 (see the Cython section below for
  why the LOCAL_Cython results are otherwise silently missing from the generated plots).

## Architecture

### Grid and graph are parallel implementations of one rail-network abstraction

`flatland/core/` defines the rail network purely in terms of generic abstractions, each with two independent
concrete implementations — grid-based (`flatland/envs/grid/`) and graph-based (`flatland/envs/graph/`). Neither
is a layer on top of the other; they're alternative representations selected at env-construction time. A
**entry point** is each representation's network-position primitive: `((row, col), direction)` for grid,
a node-id string for graph.

- **`TransitionMap`** (`core/transition_map.py`) — the rail topology: `get_transitions`/`set_transitions`,
  `get_successor_entry_points`/`get_predecessor_entry_points`/`is_valid_entry_point`. Grid impl:
  `GridTransitionMap` → `RailGridTransitionMap`. Graph impl: `GraphTransitionMap`.
- **`ResourceMap`** (`core/resource_map.py`) — maps an entry point to the "resource" (occupancy unit) used for
  conflict detection. Grid's `GridResourceMap` normally resolves to `(row, col)`, but for level-free crossings
  returns `(position, direction % 2)` so the two crossing axes count as distinct resources. Graph's
  `GraphResourceMap` is identity on the node id.
- **Distance-map subsystem** (`core/entry_point_distance_map.py`, `core/distance_map.py`,
  `core/distance_map_walker.py`) — `EntryPointDistanceMap` (agent-agnostic, keyed by
  `(source_config, target_config)`) → `AgentSourceTargetDistanceMap` (agent-handle-aware, shared template-method
  `_compute()`) → concrete `DistanceMap` (grid, numpy array) / `GraphDistanceMap` (graph, nested dict).
  `DistanceMapWalker` does the backward BFS that fills it in, one independent walk per target entry point
  (a shared visited set across targets silently breaks on cyclic/looped layouts). `RailEnv` keeps one
  `distance_map` instance alive across `reset()` calls rather than reconstructing it per episode.

### `flatland/envs/` — env control flow

`RailEnv`/`AbstractRailEnv` (`rail_env.py`) is the env facade, generic over `(TransitionMap, ResourceMap,
EntryPoint)`. `reset()` calls `rail_generators.py` (topology), `line_generators.py` (agent start/target
assignment), `timetable_generators.py` (departure/arrival windows). Per `step()`, for each agent: derive the
desired next entry point from the action via the `step_utils` state machine (`TrainState`/
`TrainStateMachine`); look up both agents' current/next *resources* via `resource_map.get_resource(...)`; feed
`(current_resource, new_resource)` pairs into `agent_chains.py`'s `MotionCheck`, which resolves cross-agent
conflicts (head-on swaps, same-target collisions) once all agents for the step are registered; then finalize
state/position, then `handle_done_state()`, then rewards, per agent, in that order. `EnvAgent` (`agent_utils.py`)
holds per-agent state; `observations.py`/`predictions.py` build the observation returned to policies, typically
via the distance map's shortest paths.

`handle_done_state()` running *before* `rewards.step_reward()` matters: it sets `agent.target_entry_point`
and, if `remove_agents_at_target` (the default), clears `agent.current_entry_point` to `None` — so on the
exact step an agent reaches `TrainState.DONE`, a `Rewards.step_reward()` implementation already sees
`current_entry_point is None`. A reward implementation that needs "where is this agent right now" must key
off `target_entry_point` (or `agent_utils.virtual_entry_point()`) for a `DONE` agent, not
`current_entry_point` — `rewards.py`'s `PunctualityRewards` missed this once and silently dropped a
departure booking as a result.

`flatland/envs/graph_rail_env.py`'s `GraphRailEnv(AbstractRailEnv[GraphTransitionMap, GraphResourceMap, str])`
is a full graph-native sibling to `RailEnv`, not just an implementation detail of the grid/graph split above.
`GraphRailEnv.from_rail_env()` converts an existing grid `RailEnv` into its graph-native equivalent (via
`GraphTransitionMap.grid_to_digraph`), while `GraphRailEnv.from_graph()` builds one directly from an
`nx.DiGraph` plus string-node-id `agent_waypoints` — no grid tuples or `Waypoint` objects involved at all. Both
envs share `AbstractRailEnv.step()`/`handle_done_state()` verbatim (a single shared definition), so behavior
differences between the two are almost entirely about topology/entry-point representation, not control flow.

### Entry point values are sanitized against numpy-dtype taint

`agent_utils.py`'s `_sanitize_entry_point()` coerces a grid `(position, direction)` entry point's numpy
scalar elements (e.g. `np.int64` from rail/line generation) to plain `int` — left untouched, this numpy-ness
can later break tuple equality (e.g. `agent_chains.py`'s level-free-crossing resource comparisons raising "the
truth value of an array with more than one element is ambiguous"). It's wired in as an attrs `converter=` on
`EnvAgent`'s four entry-point attribs (`initial_entry_point`/`current_entry_point`/`old_entry_point`/
`target_entry_point`), but attrs converters only run in `__init__` — any direct assignment after construction
(e.g. `agent.current_entry_point = ...` in `rail_env.py`'s `step()`) must call `_sanitize_entry_point()`
explicitly itself, unless the assigned value is already a known-sanitized entry point copied from elsewhere
on the same agent.

### Speed/distance are `Fraction`s, and `None` while off map

`SpeedCounter` (`envs/step_utils/speed_counter.py`) stores `max_speed` as a `Fraction` unconditionally, but
`speed`/`distance` are `Optional[Fraction]`: both `None` until the agent enters the map
(`TrainState.is_off_map_state()`: `WAITING`/`READY_TO_DEPART`/`MALFUNCTION_OFF_MAP`), and both set back to
`None` the instant it leaves it. Map entry itself flows through the same per-step `SpeedCounter.step()` call as
every other step - `rail_env.py`'s "(10b) SPEED_COUNTER UPDATE" bootstraps `distance` to `0` and accelerates
from `0` by `acceleration_delta` immediately on the step the agent departs, rather than snapping straight to
`max_speed`. `_pseudo_fractional()` snaps any `int`/`float`/`Decimal` input to a `Fraction` on the way in
(including a "nice fraction" heuristic, e.g. `0.33 -> Fraction(1, 3)` within tolerance). `RailEnv.__init__`'s
`acceleration_delta`/`braking_delta` default to `Fraction(1)`/`-Fraction(1)` to match. Passing a plain `float`
for either instead (as opposed to a `Fraction`) can silently reintroduce floats into `new_speed` mid-`step()`,
since `Fraction + float` coerces to `float` in Python — this can violate the `Fraction`-only invariant assumed
by `_cap_speed`'s `assert isinstance(v, Fraction)` for any delta that doesn't saturate to a speed boundary (0
or `max_speed`) in one step.

A pickle predating this design can carry a stale off-map `speed` pinned at `max_speed` (instead of `None`) -
`agent_utils.py`'s `load_env_agent()` normalizes a loaded agent's `speed_counter` against its `TrainState` on
load (off map → `None`/`None`; on-map `MALFUNCTION` → `speed=0`, distance left untouched) to guard against this.

`SpeedCounter` also exposes four static, `lru_cache`d formula methods (`speed_after_acceleration`,
`speed_after_braking`, `distance_after_crossing`, `distance_without_crossing`) that are the single source of
truth for the accel/brake/crossing math, all `None` (off map) in, `None` out - used both by `SpeedCounter`
itself (`step()`/`_distance_update`) and by `rail_env.py`'s post-step invariant checks to verify the actual
post-step value against the same formula. Change the math in one place, not independently in `step()` and in
the invariant that checks it.

### Step pre/post-condition assertions (`check_step_pre_post_conditions`)

`AbstractRailEnv.step()` itself calls `_check_pre_step_invariants_and_capture_snapshot()` up front, and
`_check_malfunction_state_invariant()` / `_verify_mutually_exclusive_resource_allocation()` /
`_check_post_speed_distance_invariants()` / `_check_post_position_invariants()` (`rail_env.py`) at the end - a
correctness net verifying every per-agent speed/distance/position update and resource allocation this step
matches exactly what the action/state-machine/motion-check outcome implies, not part of normal control flow.
Since these live in `AbstractRailEnv` itself (not a subclass override), `RailEnv` and `GraphRailEnv` both run
them identically - `RailEnv.step()`'s own override only adds `_update_agent_positions_map()` after calling
`super().step()`. All are gated behind the `check_step_pre_post_conditions` constructor flag (default `True`)
so the extra per-step overhead can be disabled where it matters - `examples/flatland_performance_profiling.py`'s
`get_rail_env()` defaults it to `False`, since that script exists specifically to profile `step()`'s own
cumulative time and the assertions would otherwise inflate every measurement.

`_check_post_position_invariants`/`_check_post_speed_distance_invariants` branch purely on position
(`current_entry_point is None`) and resource/malfunction/action outcomes, never on `agent.state` - the
off/on-map position signal is the invariant these checks themselves help enforce, so re-deriving branches from
`agent.state` there would be redundant with (and could drift from) position. `_check_malfunction_state_invariant`
is the one exception, since it specifically checks state/malfunction-counter consistency.

### Cython-accelerated hot paths (`ext-modules`)

A handful of hot-path modules are compiled with Cython **by default**, declared in `pyproject.toml`'s
`[tool.setuptools] ext-modules` (currently `flatland/envs/step_utils/state_machine.py`,
`flatland/envs/step_utils/states.py`, `flatland/envs/rail_env_shortest_paths.py`) with `optional = true`.
`cython` is itself a `[build-system] requires` entry, so pip/build isolation provisions it automatically into
every build — combined with `optional = true`, this means the ext-modules compile automatically whenever a C
compiler happens to be available, with zero extra flags, and gracefully fall back to the plain-Python sources
(with a warning per module) when one isn't — see the README's "Cython-accelerated build" section for the
end-user-facing version of this, including how to force the fallback deliberately (fake a missing compiler via
`CC`/`CXX` — there's no dedicated opt-out flag).

**Only an sdist is published to PyPI — no wheel — and this is deliberate, not an oversight.** A wheel's
platform/ABI tag is decided by whether Cython ever *attempted* to cythonize a module, not by whether the final
C-compile succeeded: a wheel built with Cython present but the compiler faked missing (verified empirically
this session) still comes out tagged e.g. `cp313-cp313-linux_aarch64`, never the universal `py2.py3-none-any`
one flatland-rl has always shipped. Since `cython` is now unconditionally in `[build-system] requires`, there is
no build invocation — isolated or not — that can produce a wheel without Cython attempting to process it, so
there's no way left to get a universal-tagged wheel short of a real per-platform build matrix (`cibuildwheel`,
Windows/macOS CI from scratch — not currently done; every CI job runs on `ubuntu-24.04` only). Publishing
sdist-only sidesteps the whole problem: an sdist has no platform tag at all, so every install builds — and
compiles or falls back — on the *user's own* machine. `tox.ini`'s `[testenv:build]` (used by
`publish-pypi`/`publish-test-pypi` in `publish.yml` and `checks.yml`'s own `build` job) builds `--sdist` only
for exactly this reason.

Each accelerated module stays an ordinary, fully-interpretable `.py` file (Cython's ["pure Python
mode"](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html#augmenting-pxd)); C-level types are added
via a same-named companion `.pxd` file next to it (e.g. `rail_env_shortest_paths.pxd`), which Cython picks up
automatically at compile time and which the plain-Python interpreter ignores entirely — so a `.pxd` can declare
things (`cdef`/`cpdef` function signatures, typed memoryviews, `cdef class`) that would break plain-Python
execution if they lived in the `.py` file itself; only *local variable* typing (`var: cython.int = ...`, matching
`state_machine.py`'s style) can go directly in the `.py` body, since `.pxd` files can't declare function-body
locals. `state_machine.py` unconditionally does `import cython` and uses `cython.int`-annotated locals
regardless of whether the module ends up compiled — this works fine uncompiled too, since `cython`'s
pure-Python shadow package provides working fallback stubs for these. CI cross-checks all build outcomes via
`scripts/verify_cython_extension_build.py`, which inspects either a built wheel (`--artifact wheel`, the
default) or the published sdist (`--artifact sdist`): `verify-build-no-gcc` (Cython present, compiler faked
missing) asserts a wheel's pure-Python fallback via `--expect pure-python` — there's deliberately no
`verify-build-no-cython` equivalent, since `python -m build` (isolated or not) always mandates every
`[build-system] requires` entry be satisfiable and fails outright rather than falling back if Cython itself is
missing (`optional = true` only ever covers a missing *compiler*, confirmed the hard way — an earlier version
of this env tried `--no-isolation` with cython omitted from `deps` and just got `ERROR Missing dependencies:
cython>=3.2.9`).
`py{3.10,3.11,3.12,3.13,3.14}-verify-cython-build` (plain isolated build, zero flags) and its `-no-isolation`
sibling (mirrors `pip install --no-build-isolation -e .`) both assert a wheel's real compilation via `--expect
compiled`; `[testenv:build]` asserts the published artifact is a pure-Python `--artifact sdist` (trivially true
by construction, but guards against e.g. stray `.c`/`.so` files accidentally being packaged).

**None of the above actually execute the compiled code** - they only check that a `.so`/`.pyd` artifact exists
(or doesn't). A `.pxd`/`.py` mismatch that compiles cleanly but breaks at runtime (e.g. a `cdef class` field
assigned in the `.py` source but never declared in the `.pxd` - Cython's `cdef class` has no `__dict__`
fallback for undeclared attributes, unlike a plain Python object) previously only surfaced via the profiling
notebook's `LOCAL_Cython` step, since that's the only path that both compiles in-place and actually runs
`env.step()` against the result. `py{...}-verify-cython-build-no-isolation` now also builds in-place and runs
the tests marked `pytest.mark.cython_ext` against it for exactly this reason - see the `Commands` section above
and `CONTRIBUTING.md`'s Cython section.

**Known gap: `cProfile` can't see calls into compiled Cython functions on Python 3.12.** Python 3.12's `cProfile`
registers itself via PEP 669's `sys.monitoring` instead of the legacy `PyEval_SetProfile` hook, and Cython's own
`sys.monitoring` bridge (`CYTHON_USE_SYS_MONITORING`) is gated to Python ≥3.13 — confirmed independent of the
Cython version (reproduced identically with `cython==3.2.9` and the previously-pinned `3.3.0a1`; upgrading
Cython alone does not fix it). `@cython.profile(True)` is set on `get_k_shortest_paths` for when this gets
fixed upstream, but on 3.12 it's a no-op — this is exactly why `checks.yml`'s `profiling`/
`profiling-get-k-shortest-paths` jobs publish their gist/PR-comment results from the Python **3.13** matrix leg,
not 3.12 (see `cython/cython#5470`).

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

### Testing patterns

Most state-machine/speed/malfunction behavior tests (`tests/test_multi_speed.py`, `tests/test_variable_speed.py`,
`tests/test_flatland_malfunction.py`, etc.) drive the env through `tests/test_utils.py`'s `Replay`/`ReplayConfig`
(attrs classes) via `run_replay_config()`, rather than asserting after ad-hoc `env.step()` calls. Each `Replay`
entry declares the expected `position`/`direction`/`state`/`speed`/`distance`/`malfunction` to verify *before*
that step, then the `action` to apply - so a test reads as a step-by-step table of expected states rather than
imperative code. `skip_reward_check`/`skip_action_required_check` opt out of the (fragile) reward/action-required
assertions when a test only cares about position/speed/state progression.

For a specific env state that's tedious or fragile to reach by scripting actions from scratch (e.g. deep into a
multi-agent malfunction/deadlock scenario), prefer capturing a one-off snapshot via `RailEnvPersister.save()` once
and loading it in the test via `RailEnvPersister.load_new()`, rather than replaying a long action script - the
latter is brittle against unrelated timing changes elsewhere in the env (see `tests/test_known_flatland_bugs.py`'s
`test_two_trains_on_same_cell_bug_FIXED` and its committed `*_snapshot.pkl` fixture for the pattern).

#### Docstrings for tests that step `RailEnv` and assert on agent state

A test that drives a `RailEnv` through `step()` and asserts on `agent.state`/`agent.speed_counter`/
`agent.current_entry_point` should have a docstring that describes the scenario purely in the domain vocabulary
a *user* of `RailEnv` would use - agent speeds, distances, positions/cells, and states/actions - not in terms of
how `step()`, `TrainStateMachine`, or `MotionCheck` internally arrive at that outcome. Concretely:

- Open with a one- or two-sentence scenario line naming the agents, the concrete cells/positions involved (e.g.
  "L=(3,8) and R=(3,7)"), and any shared parameter (max speed, a parametrize dimension) - not an abstract
  description.
- Follow with a flat bullet list, one bullet per phase of the scenario in the order it happens (a `- Setup:` bullet
  is almost always first). State speed/distance/state values as the reader would observe them via
  `agent.speed_counter.speed`/`.distance`/`agent.state`/`agent.current_entry_point`, not via the internal signals
  that produce them (`candidate_speed`, `resource_check`, `movement_allowed`, `crossing_completed`, etc.) - those
  belong in an inline comment next to the specific assertion they explain, not the docstring.
- Each bullet states what happens and what the test expects/observes at that phase, so the docstring alone tells
  you what the test verifies without reading the body - e.g. "A tries to cross into R - denied: position stays on
  L, speed drops back to 0", not "we then check that A's position doesn't change".
- Never narrate the act of writing or discovering the test itself - no "confirmed", "verified", "also caught and
  documented", "discovered by running it", or similar. State the behavior as fact; if a value was surprising or
  needed to be checked against a real run rather than derived by hand, that belongs in conversation with whoever
  asked for the test, not in the docstring.
- A non-obvious setup trick (e.g. why a blocking agent needs a reduced max speed to avoid completing an in-flight
  crossing before it can be braked) gets its own bullet or an inline comment at the point it matters, phrased as
  the fact itself ("the leader is exactly at its own boundary ... so that crossing is already in flight and still
  completes"), not as a note about the test author's process.
- See `test_platoon_all_stop_together_once_leader_stops_and_stays_stopped` and
  `test_agent_blocked_at_boundary_cannot_accelerate_nor_advance_into_stopped_neighbor` in
  `tests/test_flatland_envs_rail_env.py` for the target shape.

### Other top-level dirs

- `examples/` — standalone runnable scripts (custom observations, custom rail maps, training, the
  `flatland_performance_profiling.py` script driven by `benchmarks/flatland_performance_profiling.ipynb`) - not
  part of the installed package, not covered by the core test suite.
- `callbacks/` — episode-lifecycle hooks (e.g. movie generation).
- `evaluators/` — AIcrowd competition evaluation service/client + trajectory-based evaluators.
- `integrations/interactiveai/` — REST API client for the InteractiveAI dashboard.
- `utils/` — rendering (`rendertools.py`, `editor.py`), grid helpers, seeding.
- `env_generation/` — higher-level convenience env-builder over the generators.
- `png/`, `svg/` — static rendering image assets, not code.
- `benchmarks/` — performance profiling/regression scripts and notebooks (exercised by the `tox -e
  *-profiling*`/`*-benchmarks` envs); `benchmark_episodes.py` is itself a pytest module, so it's swept into a
  bare `pytest` invocation from the repo root, not just the dedicated benchmark envs.
- `scripts/make_coverage.py` — runs the suite under `coverage`, generates an HTML report, opens it in a browser
  (same as `tox -e coverage`).

### CLI entry points (`pyproject.toml`'s `[project.scripts]`, implemented in `cli.py`)

`flatland-demo` (smoke-test render demo) and `evaluator`/`flatland-evaluator` (→ `evaluators/service.py`'s
`FlatlandRemoteEvaluationService`, the AIcrowd competition-side evaluation service) live in `flatland/cli.py`.
The `flatland-trajectory-*` scripts (generate-from-policy/generate-from-metadata/evaluate/analysis) map to
`trajectories/policy_runner.py`, `trajectories/policy_grid_runner.py`, and `evaluators/trajectory_evaluator.py`
/`trajectory_analysis.py` respectively.

## Releases and versioning

- The package version is **not** a static field in `pyproject.toml` (`dynamic = ["version"]`) — it's derived by
  `setuptools_scm` from git tags at build time.
- **`release-please`** (`.github/workflows/publish.yml`'s `release-please` job) runs on every push to `main`,
  parses [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) messages since the last release,
  and opens/updates a release PR that bumps the version and updates `CHANGELOG.md`. Merging that PR creates the
  release tag, which triggers the same workflow's `test` → `publish-pypi` → `docker-publish` chain.
- To publish a release candidate to **Test PyPI** without cutting a real release, manually trigger
  `publish.yml` (`workflow_dispatch`) with a `version` (no leading `v` — the workflow prepends it itself for the
  Docker tag; a leading `v`/`V` is tolerated and stripped, but an empty result after stripping fails the run).
  This runs `test` → `publish-test-pypi` → `docker-publish` instead.
- PRs are squash-merged; adjust the squashed commit's subject/body to accurately describe the change, since that
  message is what `release-please` parses.

## Conventions (see `CONTRIBUTING.md` for full detail)

- Commit messages always follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
  (`type(scope): summary`, e.g. `fix(envs): ...`, `docs: ...`, `refactor(rail_env): ...`) - not just PR titles
  (see "Releases and versioning" above for why PR titles specifically matter to `release-please`); every commit
  on the branch should follow the same convention.
- Call a method/function with more than a couple of parameters using keyword arguments at the call site
  (`self._foo(action=action, state=state, ...)`), not positional - positional args for a long signature are
  easy to silently mis-order (especially several same-typed/`Optional` params in a row) and a keyword mismatch
  fails loudly instead.
- Prefer `NamedTuple` over a plain unnamed `Tuple` or `Dict` for structured data that doesn't need methods.
- Use `attrs` (`@attrs`/`attrib`) for classes that must keep multiple members in sync as an invariant.
- Use `abc.ABCMeta`/`abc.abstractmethod` for extension-point base classes.
- Docstrings follow numpydoc format; type hints are expected throughout (PEP 484).
- Avoid currying/closures to encapsulate state — prefer a class when the object needs multiple methods.
- Cython speed-ups go through `.pxd`-augmented pure-Python `.py` files, never `.pyx` — see "Cython-accelerated
  hot paths" above.
- Read packaged resource files (e.g. rail data shipped inside a subpackage) via `importlib_resources`
  (`path`/`read_binary`), not a raw path relative to the module.
- `TypeVar` names are always `T`-suffixed (`EntryPointT = TypeVar('EntryPointT')`, `DistanceMapT =
  TypeVar('DistanceMapT')`, etc., down to `ObsT`/`ActT`) - matches pylint's default `typevar-rgx`, which rejects
  a bare `T`/`T1`. The string passed to `TypeVar()` must match the variable name exactly. This also sidesteps a
  real footgun: several of these names (`TransitionMapT`, `ResourceMapT`, `TransitionsT`) would otherwise
  collide with an imported class of the same bare name used elsewhere in that same file (e.g. `rail_env.py`
  imports both `TransitionMap` and `ResourceMap` as classes) - a bare-named TypeVar declaration there would
  silently shadow the import for every reference below it in the file.
- When a `TypeVar` has a `bound=`, its name must be exactly `<BoundClassName>T` (e.g. `policy.py`'s
  `EnvironmentT = TypeVar('EnvironmentT', bound=Environment)`, `rail_env_policy.py`'s `RailEnvT`/
  `RailEnvActionsT` bound to `RailEnv`/`RailEnvActions`, `distance_map_walker.py`'s `EntryPointDistanceMapT`
  bound to `EntryPointDistanceMap`) - an abbreviated or generic name (e.g. a bare `EnvT` bound to `Environment`)
  is disallowed even though it's still `T`-suffixed, since the point is that the name alone should tell you the
  bound without checking the declaration. An *unbound* `TypeVar` has no such constraint and can stay generic
  (e.g. `env_observation_builder.py`'s unbound `EnvT`, `entry_point_distance_map.py`'s unbound `DistanceMapT`) -
  don't confuse the two: the same base name can be legitimately unbound-and-generic in one file and
  bound-and-specific in another.
- In comments/docstrings, always be precise about which *switch type* is meant - symmetric vs. single
  (`RailEnvTransitionsEnum.symmetric_switch_from_*` vs. the single/ordinary switch transitions) - since the two
  differ in which actions are valid at them and this distinction matters in most cases (e.g. a symmetric switch
  makes `MOVE_FORWARD` invalid straight through, where a single switch would accept it). If "switch" is used
  unqualified to mean any type, say so explicitly, e.g. "switch (of any type)".
