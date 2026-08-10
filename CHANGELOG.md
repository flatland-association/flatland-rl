# Changelog

All notable changes to Flatland will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [4.3.0](https://github.com/flatland-association/flatland-rl/compare/v4.2.6...v4.3.0) (2026-08-10)


### Features

* **backend:** fix edge case in link map. ([477b564](https://github.com/flatland-association/flatland-rl/commit/477b56458a61c68b82a6c3a0d1d8d19619cefc2a))
* **baseimage:** add base image. ([#456](https://github.com/flatland-association/flatland-rl/issues/456)) ([6992491](https://github.com/flatland-association/flatland-rl/commit/69924913baf015208b28b67d205e8031b0e3cf59))
* **chore:** add is deadend/straight/one_one for rail env transitions. ([274275b](https://github.com/flatland-association/flatland-rl/commit/274275b51af5dc5782596142d353107c07e1b821))
* fix case when the mapped cells are not neighbors. ([909efa2](https://github.com/flatland-association/flatland-rl/commit/909efa2cc00931dc24f9ed88bda2727acb6bf451))
* **grid4-utils:** add find_connected_cells flood-fill from an open set. ([8d5a723](https://github.com/flatland-association/flatland-rl/commit/8d5a723ead4ec5bc5d43b86ff5cb79127ffa7adc))
* handle double slips in link map. ([e7c3a59](https://github.com/flatland-association/flatland-rl/commit/e7c3a5920c2226a3dc85950de3fc2ed8d604ddf2))
* interface for exposing stations and inter-city lines in rail env and rail generators. ([#441](https://github.com/flatland-association/flatland-rl/issues/441)) ([0d341ce](https://github.com/flatland-association/flatland-rl/commit/0d341cee5e750a7467b4ac5a0515d4ef9cbbad92))
* **policy runner:** add possibility to change the policy with corresponding obs builder. ([890b687](https://github.com/flatland-association/flatland-rl/commit/890b6872ee31a334b7d1d9c867e96e8dcf7437a2))
* **rail generator:** expose stations and inter-city lines. ([d826504](https://github.com/flatland-association/flatland-rl/commit/d8265048e82e2e02705950a790831801ce0ae5cd))
* **rail-generator:** ban inter-city fibre search from cutting through inner-city tracks. ([fef9f0e](https://github.com/flatland-association/flatland-rl/commit/fef9f0e3c2a4ca842745465adee099d8ec7c0680))
* **rewards:** add delay rewards. ([#486](https://github.com/flatland-association/flatland-rl/issues/486)) ([6e395f8](https://github.com/flatland-association/flatland-rl/commit/6e395f8c41faf1fbbd0e39d0a2ec958955452e14))
* **rewards:** add ecml2026 fine-grained-rewards. ([#454](https://github.com/flatland-association/flatland-rl/issues/454)) ([15998d4](https://github.com/flatland-association/flatland-rl/commit/15998d4f42018e4930dc9838918205fced195368))


### Bug Fixes

* __getstate__ crashes with AttributeError when condition is a plain callable. ([ced02bc](https://github.com/flatland-association/flatland-rl/commit/ced02bc8734a3769bc69b9e3ace9932cdb23dddc))
* **ci:** verify CI-env image is pullable, not just that its manifest … ([#494](https://github.com/flatland-association/flatland-rl/issues/494)) ([0854483](https://github.com/flatland-association/flatland-rl/commit/0854483c3475435dabb2e376051e7f4f0cf3c77b))
* **evaluator-service:** supported client versions fails if flatland is not installed in env. ([#455](https://github.com/flatland-association/flatland-rl/issues/455)) ([bf6e5f2](https://github.com/flatland-association/flatland-rl/commit/bf6e5f2dd784edebc01d7a935056a2498e363196))
* global obs gym. ([ec9c1c7](https://github.com/flatland-association/flatland-rl/commit/ec9c1c78155f9ad55e0650b0907feea5ffdd69bb))
* global obs gym. ([26de6ca](https://github.com/flatland-association/flatland-rl/commit/26de6ca645212715af0e43d378e3527a857a5395))
* **persistence:** combine effects_generator override in set_full_state instead of discarding it ([c61f68a](https://github.com/flatland-association/flatland-rl/commit/c61f68a7ca398754e93629d9785a48f47a9e07a7))
* **persistence:** keep serialized agent targets hashable for older readers ([3b68a29](https://github.com/flatland-association/flatland-rl/commit/3b68a29e177e88d240eb2167dbad8d0a78fcdb88))
* **persistence:** keep serialized agent.targets hashable for older readers. ([#491](https://github.com/flatland-association/flatland-rl/issues/491)) ([3b68a29](https://github.com/flatland-association/flatland-rl/commit/3b68a29e177e88d240eb2167dbad8d0a78fcdb88))
* **persistence:** replace effects_generator instead of combining it, matching obs_builder/rewards. ([394e3fe](https://github.com/flatland-association/flatland-rl/commit/394e3fede4101c7a85e383d9bcb7f6d896016866))
* **persistence:** restore dev_obs_dict guarded by its own presence check. ([399e184](https://github.com/flatland-association/flatland-rl/commit/399e1848f737d897919c2d8d042c53d7ce1cbea0))
* **policy-runner:** cleanup. ([582a607](https://github.com/flatland-association/flatland-rl/commit/582a6078c63360484971aa428c5332f1fe8192e9))
* **policy-runner:** CLI fork uses source's ep_id instead of user's. ([0429a37](https://github.com/flatland-association/flatland-rl/commit/0429a37f8850f9e3bb1008e84dd1cd3a8959c78a))
* **policy-runner:** restore clear error and regression test for env/trajectory step mismatch. ([f6c13fa](https://github.com/flatland-association/flatland-rl/commit/f6c13fa46afcf493bf4c205771e435c3a5f95b89))
* py310 numpy dtype. ([#492](https://github.com/flatland-association/flatland-rl/issues/492)) ([b41a964](https://github.com/flatland-association/flatland-rl/commit/b41a9643eb871e50c5bd4be337b25add15d69ad1))
* **rail-generator:** fix city naming overflow and drop redundant path search. ([69cab66](https://github.com/flatland-association/flatland-rl/commit/69cab665ec74f4c9465ad03768f5fdca113dadd6))
* **rail-generator:** replace remaining debug print with warnings.warn. ([cd53b3e](https://github.com/flatland-association/flatland-rl/commit/cd53b3ecaf6c7cdd805fe7930b12fc39a470b315))
* **rail-generator:** skip empty/mismatched inter-city connections. ([cbad3c6](https://github.com/flatland-association/flatland-rl/commit/cbad3c68fd9f0c1cd14cb7041e289de16ef25c99))
* **rewards:** collision penalty should not apply when controller issues the `STOP` action. ([#452](https://github.com/flatland-association/flatland-rl/issues/452)) ([27b4c4b](https://github.com/flatland-association/flatland-rl/commit/27b4c4bb3860eab3106a81160119df2c3d1fa7d2))
* **rewards:** intermediate stop served if train stops at any halting cell of the station ([#453](https://github.com/flatland-association/flatland-rl/issues/453)) ([7db80f3](https://github.com/flatland-association/flatland-rl/commit/7db80f34cd06771a4a439fbb81fb3e867ef7cdf4))
* **rewards:** normalization BaseDefaultRewards affecting also BaseECML2026Rewards. ([#457](https://github.com/flatland-association/flatland-rl/issues/457)) ([a4d5b0a](https://github.com/flatland-association/flatland-rl/commit/a4d5b0a7420d79f8b9cdc52e4375f2d3f69c6cb3))
* **sparse rail gen:** one link per gate-gate pair according to data model. ([#451](https://github.com/flatland-association/flatland-rl/issues/451)) ([4da01ed](https://github.com/flatland-association/flatland-rl/commit/4da01ed908df69540d3b22f340959ead464fd613))
* **tests:** de-flake test_env_generator_no_seed. ([#493](https://github.com/flatland-association/flatland-rl/issues/493)) ([1ad097f](https://github.com/flatland-association/flatland-rl/commit/1ad097f05750e7e915b4e6e5a731fbdb2b290781))
* **trajectories:** replay-to-step fallback drops obs_builder/rewards/effects_generator overrides. ([d0f92a0](https://github.com/flatland-association/flatland-rl/commit/d0f92a078b07e604f8c5f81b9ba120f875c80437))
* Zero-arg construction defers a TypeError to deep inside env.step(). ([8c363af](https://github.com/flatland-association/flatland-rl/commit/8c363af65b7327b24ebd51db9e32570c906d1a26))


### Performance Improvements

* **core:** add Cython to performance profiling. ([191c30b](https://github.com/flatland-association/flatland-rl/commit/191c30ba6cb07f7ec423f921aa86d8d8d6557fa6))
* **core:** profile get-k-shortest-paths. Define performance threshold for step profiling. ([#448](https://github.com/flatland-association/flatland-rl/issues/448)) ([2cc09ca](https://github.com/flatland-association/flatland-rl/commit/2cc09ca48c9a111adf798f5c70a78b1cf168bb69))
* **core:** use Cython for state machine and states. ([#449](https://github.com/flatland-association/flatland-rl/issues/449)) ([b70137c](https://github.com/flatland-association/flatland-rl/commit/b70137cb4206ac5595e67ded4007c03ddaed6cf2))
* **rewards:** use dict.from_keys from tuple instead of comprehension from set. ([#484](https://github.com/flatland-association/flatland-rl/issues/484)) ([90359df](https://github.com/flatland-association/flatland-rl/commit/90359df7775ac20037a11f138c718f660784704a))
* **shortest paths:** use default dict to avoid greedy initialization. ([c40874f](https://github.com/flatland-association/flatland-rl/commit/c40874fec0e7d4f0990204b35dc894feaf42ef8e))
* **shortest paths:** use heap per length to avoid iterating through all paths in the heap. ([997898b](https://github.com/flatland-association/flatland-rl/commit/997898b3dc3d52134284ba983f2173ebf64ba16d))


### Miscellaneous Chores

* release 4.2.7 ([4c58875](https://github.com/flatland-association/flatland-rl/commit/4c58875c8a42109802fe68a08d2b13b9f1533bc4))
* release 4.3.0 ([9b1b014](https://github.com/flatland-association/flatland-rl/commit/9b1b014aba17aacf297d4a43a531ee5523aa14c9))

## [[v4.2.6]](https://github.com/flatland-association/flatland-rl/compare/v4.2.5...v4.2.6) 2026-06-01

### Added

* feat(cli): allow all click commands to be run as module. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/430

### Changed

* chore(docs): prepare release notes v4.2.5. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/416
* chore(deps-dev): bump jupyter-server from 2.17.0 to 2.18.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/417
* chore(deps): bump gitpython from 3.1.46 to 3.1.47 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/411
* chore(deps-dev): bump jupyterlab from 4.5.2 to 4.5.7 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/414
* chore(deps-dev): bump notebook from 7.5.2 to 7.5.6 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/413
* chore(deps): bump urllib3 from 2.6.3 to 2.7.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/421
* chore(deps-dev): bump mistune from 3.2.0 to 3.2.1 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/418
* chore(deps): bump gitpython from 3.1.47 to 3.1.50 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/420
* chore(deps): bump idna from 3.11 to 3.15 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/422
* chore: pin redis<8. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/431

### Fixed

* perf(core): cache fraction comparisons. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/427
* perf(core): add caching speed counter and rewards. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/428
* perf(core): improve performance of shortest path finding by caching `Waypoint` hashes. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/426

## [[v4.2.5]](https://github.com/flatland-association/flatland-rl/compare/v4.2.4...v4.2.5) 2026-05-1

### Added

* feat: pessimistic action required. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/401
* feat(rewards): add minimum penalty for target not reached. by @CleverManu in https://github.com/flatland-association/flatland-rl/pull/397
* feat: ecml2026 competition reward by @CleverManu in https://github.com/flatland-association/flatland-rl/pull/402
* feat(RailEnvPersister): allow persistence of several malfunction generators (2). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/398
* feat(RailEnvPersister): generalize effects generator serialization (3). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/407

### Changed

* chore: add macos system file to gitignore. by @manuschn in https://github.com/flatland-association/flatland-rl/pull/392
* chore(deps): bump aiohttp from 3.13.3 to 3.13.4 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/395
* chore(deps): bump requests from 2.32.5 to 2.33.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/390
* chore(deps): bump pygments from 2.19.2 to 2.20.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/393
* chore(deps): bump pillow from 12.1.1 to 12.2.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/399
* chore(deps-dev): bump pytest from 9.0.2 to 9.0.3 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/400
* chore(deps): bump lxml from 6.0.2 to 6.1.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/405
* chore(deps-dev): bump python-dotenv from 1.2.1 to 1.2.2 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/404
* chore(deps-dev): bump nbconvert from 7.17.0 to 7.17.1 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/403
* refactor(RailEnvPersister): cleanup malfunction generator deserialization (1). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/406

### Fixed

* fix: line and timetable generators from file. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/408
* fix: evaluation of ECML2026Rewards. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/410
* fix: normalization of ECML2026Rewards. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/412

## [[4.2.4]](https://github.com/flatland-association/flatland-rl/compare/v4.2.3...v4.2.4) 2026-03-19

### Added

* feat: ignore paths leading out of the grid in `get_k_shortest_paths`. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/347
* feat(graph): make actions on graph independent of underlying graph (graph generalization part 1). by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/317
* feat(graph): refactor abstract and graph rail env (graph generalization part 2). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/351
* feat(graph): generalize env step to be grid-agnostic (graph generalization part 3). by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/353
* feat(graph): generalize distance map (graph generalization part 4). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/356
* feat(graph): partially fix graph transition map for symmetric switches (graph generalization part 5). by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/361
* feat(graph): inject malfunction generator in `from_rail_env` (graph generalization part 6). by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/365
* feat(rewards): add base class for fine-grained default rewards. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/357

* feat(rewards): support for multiple visits. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/377
* feat(persistence): persist level free crossings. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/381
* feat(graphics): add visual element level free. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/380
* feat(rendering): render all station cells not just targets. by @CleverManu in https://github.com/flatland-association/flatland-rl/pull/374
* feat: departure malfunction generator. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/383

### Changed

* chore: update dependencies, comply with upstream DLA. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/349
* chore(deps): bump protobuf from 6.33.4 to 6.33.5 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/350
* chore: update dependencies, comply with upstream DLA. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/352
* ci: increase number of retries and retry delay for notebooks. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/355
* ci: pin setuptools for running profiling on older Flatland versions. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/359
* chore(deps-dev): bump nbconvert from 7.16.6 to 7.17.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/358
* chore: update dependencies, comply with upstream DLA (part 3). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/354
* chore(deps): bump pillow from 12.1.0 to 12.1.1 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/360
* ci: bump supercharge/redis-github-action@1.7.0 to 1.8.1. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/362
* ci: disable notebooks-3.10 as failing too often. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/369
* ci: fix missing env py3.13-notebooks-no-pickle. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/371
* docs: parameters docs from init to class-level in order to fix Sphinx rendering. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/370
* refactor: drop redundant get valid move actions. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/372
* chore: add verbose flag for trajectory analysis. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/373
* test: do not skip reward comparison any more. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/375
* chore: bump deprecated gha checkout and setup-python versions to v6. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/385
* chore(deps-dev): bump tornado from 6.5.4 to 6.5.5 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/384
* chore(deps): bump pyasn1 from 0.6.2 to 0.6.3 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/387
* refactor: generators receive random state from env only, deprecate random-stateful generators. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/341

### Fixed

* fix: use fractional speed and distance. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/376

## [[4.2.3]](https://github.com/flatland-association/flatland-rl/compare/v4.2.2...v4.2.3) 2026-01-30

### Added

* feat(trajectory API): allow combination --seed with --env. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/299
* feat(Trajectory API): add cli option for callbacks to policy runner. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/305
* feat(trajectory API): statistical analysis of trajectories. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/303
* feat: add optional direction at target for `get_k_shortest_paths`. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/344
* feat: add cli options for callbacks to policy grid runner cli. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/340
* feat: add cutoff option to k shortest paths. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/345
*

### Changed

* chore(deps-dev): bump jupyterlab from 4.4.6 to 4.4.8 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/293
* refactor(env generator): allow for seed None in env_generator. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/300
* refactor(dla): avoid env in dla policy init, get it from observation. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/297
* docs: fix status badge. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/309
* refactor(trajectories API): simplify policy runner API. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/307
* feat(trajectory API): data analysis cli. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/308
* refactor(trajectory API): allow policy and observation builder to be passed through env var. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/311
* refactor(trajectory analysis): extract method. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/315
* ci: add testcontainers dev dependency for DLA test. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/320
* ci: free disk space before running tests. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/324
* chore(deps): bump urllib3 from 2.5.0 to 2.6.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/323
* chore(deps): bump fonttools from 4.59.1 to 4.61.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/322
* chore(deps): update pip dependencies. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/295
* chore(deps): bump pyasn1 from 0.6.1 to 0.6.2 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/339
* chore(deps-dev): bump wheel from 0.45.1 to 0.46.2 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/343
* refactor(trajectoy API): rename colum`normalized_reward`. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/342

### Fixed

* !fix(ml-RLlib): RLlib compatibility: no rewards after (single) agent is done. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/302
* fix(env-generator): avoid division by zero. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/306
* fix(trajectories API): fix float comparison. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/310
* fix(policy runner): fix missing reset without re-generating in policy runner. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/312
* fix(policy runner): regression if `--env-path` and `--seed` is provided according to poliy runner cli description. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/313
* fix(policy runner, evaluator callbacks): fix computation of `normalized_reward` in evaluator callback and and policy runner. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/314
* perf(Trajectory Evaluator): Pandas lookups are very slow, add caching. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/318
* fix: ignore failing on invalid initial direction. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/319
* fix: ignore failing on invalid initial direction. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/321
* fix: correct waypoint tracking in DefaultRewards by @florath in https://github.com/flatland-association/flatland-rl/pull/328
* fix: directions at intermediate waypoints in sparse line generator. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/335
* fix: prevent duplicate reward calculation for completed agents by @florath in https://github.com/flatland-association/flatland-rl/pull/329

## [[4.2.2]](https://github.com/flatland-association/flatland-rl/compare/v4.2.1...v4.2.2) 2025-09-26

### Added

* feature: offline evaluation. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/272
* feature: add routing flexibility to intermediate waypoints. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/271
* feat(policy runner): additional options and fixes. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/276

### Changed

* feat: Use GraphTransitionMap with RailEnv. refactor: Use Generic Type Hints for the Core / Envs Levels. Pull-up to core of several components. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/257
* perf(trajectories): improve collecting during policy run. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/283
* test: verify known flatland bugs from maze release 2 are fixed. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/273
* ci: run checks workflow upon push to main. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/288
* chore: update dependency post merge flatland baselines by @chenkins in https://github.com/flatland-association/flatland-rl/pull/289
* ci: fix condition run benchmarks on main. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/290
* chore: Release Notes 4.2.2. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/287
* ci: fix tests in publish workflows. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/291

## [[4.2.1]](https://github.com/flatland-association/flatland-rl/compare/v4.2.0...v4.2.1) 2025-09-03

### Added

* feature: Shortest Path Policy. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/268
* feature: basic support for Multi-Objective Reinforcement Learning (MORL). by @chenkins in https://github.com/flatland-association/flatland-rl/pull/269

## [[4.2.0]](https://github.com/flatland-association/flatland-rl/compare/v4.1.4...v4.2.0) 2025-08-29

### Added

* feature: Support version ranges in evaluator service. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/261
* feature: Add observation perturbations. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/255
* feature: Minor disruptions with faster recovery times at train stations. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/263

## [[4.1.4]](https://github.com/flatland-association/flatland-rl/compare/v4.1.3...v4.1.4) - 2025-08-16

### Added

* feature: add clone_from other env. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/249
* feature: Trajectory rollout with rllib checkpoint. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/253
* feature: Improve Trajectory API fork: load the latest snapshot and run from there instead of from beginning. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/254

### Changed

* Bump urllib3 from 2.4.0 to 2.5.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/233
* Bump pillow from 11.2.1 to 11.3.0 by @dependabot[bot] in https://github.com/flatland-association/flatland-rl/pull/246
* refactor: malfunction generation as effects generator. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/232
  and https://github.com/flatland-association/flatland-rl/pull/258
* refactor: `Policy.act` should not require handle. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/250
* Cache random state to generate cached random values instead of random values themselves. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/256

## [[4.1.3](https://github.com/flatland-association/flatland-rl/compare/v4.1.2...v4.1.3)] - 2025-06-20

### Added

* Enhance Trajectory Runner API: clone/fork trajectory and run policy from intermediate step. Add rewards and dones to Trajectory API and regression tests, add
  callback for observation and info_dict snapshoting. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/222
* Add retries to tox notebooks env. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/227
* 142 verify required and missing requirements. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/225
* Bump torch from 2.7.0 to 2.7.1 by @dependabot in https://github.com/flatland-association/flatland-rl/pull/229
* Bump requests from 2.32.3 to 2.32.4 by @dependabot in https://github.com/flatland-association/flatland-rl/pull/231

### Fixed

* Bugfix action serialization in Trajectory API. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/226
* Fix path type conversion trajectory API cli. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/228

## [[4.1.2](https://github.com/flatland-association/flatland-rl/compare/v4.1.1...v4.1.2)] - 2025-05-23

### Added

* Add 4.1.1 to list of versions to be profiled. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/212
* 116 Add support for py 3.13 by @chenkins in https://github.com/flatland-association/flatland-rl/pull/121

### Fixed

* Fix passing observation builder to RailEnv in FlatlandRemoteClient. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/215

### Changed

* Refactor action preprocessing. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/214
* Performance Tuning step function part III: TrainState/StateTransitionSignals data types and object lifecycle. by @chenkins
  in https://github.com/flatland-association/flatland-rl/pull/210
* Drop pkg_resources as it is deprecated and removed in Python 3.12. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/221

## [[4.1.1](https://github.com/flatland-association/flatland-rl/compare/v4.1.0...v4.1.1)] - 2025-05-16

### Added

* Add tqdm_kwargs to create_from_policy as well. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/198
* Add regression DLA against saved trajectories. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/208

### Fixed

* Fix is_cell_entry of SpeedCounter.__setstate__ by @castagna-a in https://github.com/flatland-association/flatland-rl/pull/199

### Changed

* Remove obsolete parameters in rewards function. Add math references. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/201
* Improved offset for agent debug. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/209
* 196-performance-step-function by @chenkins in https://github.com/flatland-association/flatland-rl/pull/203
* 86 Performance Speed-Up MotionCheck and Code/Documentation Cleanup. by @chenkins in https://github.com/flatland-association/flatland-rl/pull/87

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
