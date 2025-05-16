# Changelog

All notable changes to Flatland will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [[4.1.0](https://github.com/flatland-association/flatland-rl/compare/v4.0.6...v4.1.0)] - 2025-03-31

### Added

* 134 Add Effects Generator. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/156
* 158 Trajectory cli (runner, evaluator) improvements. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/193
* Add configuration option for non-default URL for InteractiveAI events, context and historic API. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/191

### Fixed

* Add graph_to_digraph.drawio.png required by graph demo notebook. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/189

### Changed

* 179 Simplify step function by @chenkins in https://github.com/flatland-association/flatland-rl/pull/182

## [[4.0.6](https://github.com/flatland-association/flatland-rl/compare/v4.0.5...v4.0.6)] - 2025-03-21

### Added

* Update CHANGELOG.md by @chenkins in https://github.com/flatland-association/flatland-rl/pull/159
* 125/96 Episodes with malfunction for benchmarking and regression tests. 8 Policy abstraction. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/131
* 135 Add Flatland callbacks. Refactor trajectories. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/157
* 73 Get Pettingzoo example to work again. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/102
* Bump jinja2 from 3.1.5 to 3.1.6 by @dependabot in https://github.com/flatland-association/flatland-rl/pull/155
* 172 Add check mutually exclusive cell occupation and fix step function edge cases malfunction by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/171
* 148 Simplify action preprocessing. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/163
* 148 Fix action preprocessing. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/183
* 140 Rail, Line and Timetable from File Generators. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/141
* 111 Variable Speed Profiles. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/136
* InteractiveAI Integration. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/152

### Fixed

* fix: Add tox benchmark environment again. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/167
* Fix benchmarks. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/169
* Fix graph demo visualization. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/170
* 161 Split unit tests and slow ml tests in tox for transparency and parallelism. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/164
* Fix tox gh actions. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/165
* Fix main gh wf. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/187

### Changed

* Move graph image to subfolder. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/186
* Enable checks workflow on all prs. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/160

## [[4.0.5](https://github.com/flatland-association/flatland-rl/compare/v4.0.4...v4.0.5)] - 2025-03-10

### Added

* 78 Extract performance profiling cli. by @chenkins in [pr[#88](https://github.com/flatland-association/flatland-rl/pull/88)]
* 110 Over- and underpasses (aka. level-free diamond crossings). by @chenkins in [pr[#120](https://github.com/flatland-association/flatland-rl/pull/120)]
* 109 Multi-stop Schedules (w/o alternatives/routing flexibility). by @chenkins in [pr[#124](https://github.com/flatland-association/flatland-rl/pull/124)]

### Fixed

* fix: from attr import attr, attrs, attrib, Factory ImportError: cannot import name 'attrs' from 'attr' by @chenkins
  in [pr[#127](https://github.com/flatland-association/flatland-rl/pull/127)]
* Fix grammar and spelling in comments rail_env py by @SergeCroise in [pr[#130](https://github.com/flatland-association/flatland-rl/pull/130)]
* 118 Add test_lru_cache_problem.py. by @chenkins in [pr[#119](https://github.com/flatland-association/flatland-rl/pull/119)]

### Changed

* 143 Retry for ml tests. by @chenkins in [pr[#146](https://github.com/flatland-association/flatland-rl/pull/146)]
* Enable running notebooks in main workflow by @chenkins in [pr[#144](https://github.com/flatland-association/flatland-rl/pull/144)]
* Use flatland-scenarios instead of data.flatland.cloud for trajectories. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/153

## [[4.0.4](https://github.com/flatland-association/flatland-rl/compare/v4.0.2...v4.0.4)] - 2025-02-18

### Added

* Grid to DiGraph Conversion and Graph Simplification [pr [#90](https://github.com/flatland-association/flatland-rl/pull/90)]
  and [pr [flatland-book#6](https://github.com/flatland-association/flatland-book/pull/6)]
* Policy evaluation and training cli (rllib) [pr [#85](https://github.com/flatland-association/flatland-rl/pull/85)]
    - extract ml dependencies (torch, gym, ray, etc.) to optional dependencies in `pyproject.toml`, new `requirements-ml.txt` and slimmer
      `requirements[-dev].txt`, keep core `gym` free.
    - accordingly, move corresponding code to new `flatland.ml` module
    - accordingly, move corresponding tests to `tests.ml` (tests becomes a Python module)
* Enable redis in ci and run `test_service.ipynb` in ci [pr [#65](https://github.com/flatland-association/flatland-rl/pull/65)]
* Run main workflow daily and allow for manual triggering. Update deprecated gh actions versions,
  see [GitHub Blog](https://github.blog/changelog/2024-03-07-github-actions-all-actions-will-run-on-node20-instead-of-node16-by-default/) [pr [#83](https://github.com/flatland-association/flatland-rl/pull/83)].
* Add `AWS_ENDPOINT_URL` env var to override default S3 endpoint URL in
  `aicrowd_helpers.py` [pr [#112](https://github.com/flatland-association/flatland-rl/pull/112)].
* Add episodes for benchmarking and regression tests [pr [#105](https://github.com/flatland-association/flatland-rl/pull/105)].
* Dump `results.json` (evaluation state) along `results.csv` [pr [#115](https://github.com/flatland-association/flatland-rl/pull/115)].
* Dump evaluation state along results output path. Make test env folder and supported client versions configurable for evaluation
  service. [pr [#115](https://github.com/flatland-association/flatland-rl/pull/115)].
* Policy evaluation and training cli (rllib) [pr [#85](https://github.com/flatland-association/flatland-rl/pull/85)].

### Fixed

* Add flatland-rl [apidocs](https://flatland-association.github.io/flatland-book/apidocs/index.html) back to flatland book
  again [pr [flatland-book#7](https://github.com/flatland-association/flatland-book/pull/7)]
* Fix flapping test malfunctions [pr [#103](https://github.com/flatland-association/flatland-rl/pull/103)]
* Fix `README.md` indefinite article before a vowel sound [pr [#95](https://github.com/flatland-association/flatland-rl/pull/95)]

### Changed

* Bump jinja2 from 3.1.4 to 3.1.5 [pr [#106](https://github.com/flatland-association/flatland-rl/pull/106)]
* Bump tornado from 6.4.1 to 6.4.2. [pr [#93](https://github.com/flatland-association/flatland-rl/pull/93)]
* Bump aiohttp from 3.10.10 to 3.10.11. [pr [#94](https://github.com/flatland-association/flatland-rl/pull/94)]
* Deployment [flatland-book](https://github.com/flatland-association/flatland-book) to GitHub
  Pages [pr [flaland-book#4](https://github.com/flatland-association/flatland-book/pull/4)]
  and [pr [flatland-book#5](https://github.com/flatland-association/flatland-book/pull/5)]
  and [pr [#98](https://github.com/flatland-association/flatland-rl/pull/98)]

### Removed

* Use Python >= 3.10 (drop support for deprecated python 3.8 eol 2024-10-07, see https://devguide.python.org/versions/). Move ml dependencies from core
  dependencies to optional ml dependencies [pr [#84](https://github.com/flatland-association/flatland-rl/pull/84)].
* Remove images folder. Images not referenced in documentation any more. Remove skipped test (fails tested
  locally) [pr [#82](https://github.com/flatland-association/flatland-rl/pull/82)].
* Rendering folder is not used. Descriptions in txt file are also contained in
  `examples/misc/generate_video/video_generation.md` [pr [#81](https://github.com/flatland-association/flatland-rl/pull/81)].
* Cleanup scripts folder to contain only scripts to be run with `make` [pr [#80](https://github.com/flatland-association/flatland-rl/pull/80)].
* Remove `flatland.action_plan` module as obsolete [pr [#79](https://github.com/flatland-association/flatland-rl/pull/79)]

## [4.0.3] - 2024-04-23

### Github Action Release failed

Fixed.

## [4.0.2] - 2024-04-23

### Performance improvement

The rail generators (infrastructure) implementation is based on A*. The A* implementation has been improved in terms of calculation time. The main modification
concerns the internal data structure which was widely used in the A* algorithm. The used ordered set is replaced by a heap that allows to fetch the nearest
nodes in O(1) instead of O(n).

More details: https://github.com/flatland-association/flatland-rl/pull/68

## [4.0.1] - 2023-10-30

### Fixed

- Removed dependency on an old version of `gym` which in turn brought in an old version of `pyglet` that caused issues
  on Windows.

## [4.0.0] - 2023-10-27

### Removed

- Dropped support for Python 3.7 because it's end of life.

### Changed

- Improved performance by introducing an LRU cache.
- Drastically improved performance by improving `numpy` usage.
- Updated a lot dependencies.
- Cleaned the project structure.
- First release handled by the [flatland association](https://www.flatland-association.org/)!

### Fixed

- Fixed a lot of bugs :)
- The tests actually pass now.
