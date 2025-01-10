# Changelog

All notable changes to flatland will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### New

* grid to digraph conversion. [pr [#90](https://github.com/flatland-association/flatland-rl/pull/90)]
* data structures for decision point graph. [pr [#90](https://github.com/flatland-association/flatland-rl/pull/90)]

### Removed

* Remove images folder. [pr [#82](https://github.com/flatland-association/flatland-rl/pull/82)]
* Remove rendering folder. [pr [#81](https://github.com/flatland-association/flatland-rl/pull/81)]
* Cleanup scripts folder. [pr [#80](https://github.com/flatland-association/flatland-rl/pull/80)]
* Remove action_plan module as obsolete. [pr [#79](https://github.com/flatland-association/flatland-rl/pull/79)]

### Changed

* Bump tornado from 6.4.1 to 6.4.2. [pr [#93](https://github.com/flatland-association/flatland-rl/pull/93)]
* Bump aiohttp from 3.10.10 to 3.10.11. [pr [#94](https://github.com/flatland-association/flatland-rl/pull/94)]

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
