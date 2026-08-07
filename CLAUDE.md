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

- **Run the core test suite** (matches CI's `test` job): `benchmarks/benchmark_episodes.py`'s regression tests need
  a `BENCHMARK_EPISODES_FOLDER` populated from the `FLATLAND_BENCHMARK_EPISODES_FOLDER` archive (see
  `flatland-benchmarks-episodes-url` in `checks.yml`):
  ```
  python -m pytest --ignore=tests/ml -m "not slow"
  ```
  Without that fixture set up, drop the benchmark-episode tests or just run a narrower path, e.g.
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
- **Profiling notebooks** (`benchmarks/flatland_performance_profiling.ipynb`,
  `benchmarks/benchmark_k_shortest_paths_profiling.ipynb`): `tox -e py3.13-profiling` /
  `tox -e py3.13-profiling-get-k-shortest-paths` — use Python 3.13, not 3.12 (see the Cython section below for
  why the LOCAL_Cython results are otherwise silently missing from the generated plots).

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

### Speed is always a `Fraction` internally

`SpeedCounter` (`envs/step_utils/speed_counter.py`) stores `speed`/`max_speed`/`distance` as `Fraction`
unconditionally, regardless of what type callers pass in: `_pseudo_fractional()` snaps any `int`/`float`/`Decimal`
input to a `Fraction` on the way in (including a "nice fraction" heuristic, e.g. `0.33 -> Fraction(1, 3)` within
tolerance). `RailEnv.__init__`'s `acceleration_delta`/`braking_delta` default to `Fraction(1)`/`-Fraction(1)` to
match. Passing a plain `float` for either instead (as opposed to a `Fraction`) can silently reintroduce floats
into `new_speed` mid-`step()`, since `Fraction + float` coerces to `float` in Python — this can violate the
`Fraction`-only invariant assumed by `cached_cap_speed`'s `assert isinstance(v, Fraction)` for any delta that
doesn't saturate to a speed boundary (0 or `max_speed`) in one step.

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

### Other top-level dirs

- `callbacks/` — episode-lifecycle hooks (e.g. movie generation).
- `evaluators/` — AIcrowd competition evaluation service/client + trajectory-based evaluators.
- `integrations/interactiveai/` — REST API client for the InteractiveAI dashboard.
- `utils/` — rendering (`rendertools.py`, `editor.py`), grid helpers, seeding.
- `env_generation/` — higher-level convenience env-builder over the generators.
- `png/`, `svg/` — static rendering image assets, not code.

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

- Prefer `NamedTuple` over a plain unnamed `Tuple` or `Dict` for structured data that doesn't need methods.
- Use `attrs` (`@attrs`/`attrib`) for classes that must keep multiple members in sync as an invariant.
- Use `abc.ABCMeta`/`abc.abstractmethod` for extension-point base classes.
- Docstrings follow numpydoc format; type hints are expected throughout (PEP 484).
- Avoid currying/closures to encapsulate state — prefer a class when the object needs multiple methods.
- Cython speed-ups go through `.pxd`-augmented pure-Python `.py` files, never `.pyx` — see "Cython-accelerated
  hot paths" above.
- Read packaged resource files (e.g. rail data shipped inside a subpackage) via `importlib_resources`
  (`path`/`read_binary`), not a raw path relative to the module.
