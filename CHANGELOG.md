# Changelog

All notable changes to flatland will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
