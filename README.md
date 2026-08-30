🚂 Flatland
========

![Flatland](https://i.imgur.com/0rnbSLY.gif)

[![Main](https://github.com/flatland-association/flatland-rl/actions/workflows/checks.yml/badge.svg)](https://github.com/flatland-association/flatland-rl/actions/workflows/checks.yml)

Flatland is an open-source toolkit for developing and comparing Multi-Agent Reinforcement Learning algorithms in little
(or ridiculously large!) gridworlds.

[The official documentation](https://flatland-association.github.io/flatland-book/intro.html) contains full details about the environment and problem
statement.

Flatland is tested with Python 3.10, 3.11, 3.12, 3.13 and 3.14 on modern versions of macOS, Linux and Windows. You may encounter
problems with graphical rendering if you use WSL.

🏆 Challenges
---

This library was developed specifically for the
AIcrowd [Flatland challenges](http://flatland.aicrowd.com/research/top-challenge-solutions.html) in which we strongly
encourage you to take part in!

- [ECML 2026 Challenge](https://competition.flatland.cloud/suites/6240c685-0fb4-481e-9404-47a570632227) ([documentation](https://flatland-association.github.io/flatland-book/challenges/ecml2026.html) | [starterkit](https://github.com/flatland-association/ecml2026-starterkit))
- [Flatland 3 Challenge 2021](https://www.aicrowd.com/challenges/flatland-3) ([documentation](https://flatland-association.github.io/flatland-book/challenges/flatland3.html))
- [AMLD 2021 Challenge](https://www.aicrowd.com/challenges/flatland) ([documentation](https://flatland-association.github.io/flatland-book/challenges/amld2021.html))
- [NeurIPS 2020 Challenge](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/) ([documentation](https://flatland-association.github.io/flatland-book/challenges/neurips2020-challenge.html))
- [2019 Challenge](https://www.aicrowd.com/challenges/flatland-challenge)

📦 Setup
---

### Setup virtual environment

Set up a virtual environment using your preferred method (we suggest the built-in `venv`) and activate it.
You can use your IDE to do this or by using the command line:

```shell
python -m venv .venv
source .venv/bin/activate
```

### Stable release

Install Flatland using pip:

```shell
python -m pip install flatland-rl
```

This is the preferred method to install Flatland, as it will always install the most recent stable release.

### Cython-accelerated build

A few hot-path modules are compiled with [Cython](https://cython.org/) for extra performance **automatically**,
as part of the normal install above, whenever a working C compiler is available - no extra flags needed.
Cython itself doesn't need to be installed manually either: it's a `[build-system] requires` entry in
`pyproject.toml`, so pip provisions it automatically as part of the build.

**Requirement:** a working C compiler (e.g. `gcc`/`clang` on Linux/macOS, or the Microsoft C++ Build Tools on
Windows). Compilation is optional and best-effort: if a compiler is missing, the build falls back to the
plain-Python sources instead of failing, printing a warning for each module it could not compile. Pass `-v` to
pip (`python -m pip install -v flatland-rl`) to see it - by default pip hides the underlying build output on
success. The install still succeeds either way - you get the plain-Python modules, just without the Cython
speed-up.

**Forcing a plain-Python build:** there's no dedicated flag for this - make the build think no C compiler is
available instead, the same mechanism this repo's own CI relies on to test the fallback path:

```shell
CC=/nonexistent-cc CXX=/nonexistent-cxx python -m pip install flatland-rl
```

(or `CC=/nonexistent-cc CXX=/nonexistent-cxx pip install -e .` from a checkout). This produces the same
per-module warning output as a genuinely missing compiler - expected and harmless, the install still succeeds
as pure-Python.

Only a source distribution (sdist) is published to PyPI - there's no prebuilt wheel - so this always builds
from source on your own machine. That's deliberate: a wheel with a
Cython extension gets tagged to one specific Python version/platform/ABI the moment Cython even *attempts* to
compile it, regardless of whether the compile actually succeeds - publishing one would mean every other Python
version or OS gets no matching wheel at all. Building from source sidesteps that, at the cost of needing
Python's standard packaging tools, and failing outright in environments that categorically refuse to build from
source (e.g. `pip install --only-binary=:all:`, some locked-down/air-gapped setups).

### Known issue: architecture-mismatched Cython extensions on Apple Silicon

**Symptoms:** on some x86_64-conda-on-Apple-Silicon setups (Python running under Rosetta 2), importing
flatland-rl raises something like:

```
ImportError: dlopen(.../flatland/envs/step_utils/states.cpython-312-darwin.so, ...):
tried: '...states.cpython-312-darwin.so' (mach-o file, but is an incompatible architecture (have 'arm64', need 'x86_64'))
```

even though `pip install` itself reports success.

**Diagnosis:** one or more of the Cython-accelerated modules (`flatland.envs.step_utils.states`,
`.state_machine`, `flatland.envs.rail_env_shortest_paths`) got compiled for the host chip's native
architecture (arm64) instead of the target Python's architecture (x86_64), so it fails to load.

```shell
conda info | grep platform                                    # installed conda platform (can be in mismatch with host arch): osx-64/osx-arm64
python -c "import platform; print(platform.machine())"        # arch of python on $PATH: x86_64/arm64
file "$(which python)"                                        # arch of python binary on $PATH: x86_64/arm64
uname -m                                                      # host arch (can be in mismatch with installed conda platform): arm64/x86_64
```

**Known workarounds:**

* [flatland-rl#121](https://github.com/flatland-association/flatland-rl/pull/121)
* Install an arm64-native Miniconda distribution instead of an x86_64 one - see
  [Miniconda's install guide](https://www.anaconda.com/docs/getting-started/miniconda/install).


🚀 Releases
---

* Release PRs are automatically opened
  by [release-please](https://github.com/googleapis/release-please)/[release-please-action](https://github.com/marketplace/actions/release-please-action) based
  on [Conventional Commit Messages](https://www.conventionalcommits.org/en/v1.0.0/)
* [How do I change the version number?](https://github.com/googleapis/release-please?tab=readme-ov-file#how-do-i-change-the-version-number)

👥 Credits
---

This library was initially developed
by [SBB](https://www.sbb.ch/en/), [Deutsche Bahn](https://www.deutschebahn.com/), [SNCF](https://www.sncf.com/en),
[AIcrowd](https://www.aicrowd.com/) and [numerous contributors](https://flatland-association.github.io/flatland-book/misc/credits.html) from the
flatland community. It is now developed by the [Flatland Association](https://flatland-association.org) and the [Flatland Community](https://flatland.cloud).

➕ Contributions
---
Please follow the [Contribution Guidelines](./CONTRIBUTING.md) for more details on how you can successfully contribute
to the project. We enthusiastically look forward to your contributions!

💬 Communication
---

* [Issue Tracker](https://github.com/flatland-association/flatland-rl/issues/)

🔗 Partners
---
<a href="https://sbb.ch" target="_blank" style="margin-right:30px"><img src="https://flatland-association.org/members/sbb-cff-ffs-logo.svg" alt="SBB" height="60"/></a>
&nbsp;
<a href="https://www.deutschebahn.com/" target="_blank" style="margin-right:30px"><img src="https://i.imgur.com/pjTki15.png" alt="DB"  height="60"/></a>
&nbsp;
<a href="https://www.sncf.com/en" target="_blank" style="margin-right:30px"><img src="https://iconape.com/wp-content/png_logo_vector/logo-sncf.png" alt="SNCF"  height="60"/></a>
&nbsp;
<a href="https://www.aicrowd.com" target="_blank"><img src="https://i.imgur.com/kBZQGI9.png" alt="AIcrowd"  height="60"/></a>
&nbsp;
<a href="https://flatland.cloud" target="_blank"><img src="https://flatland-association.org/members/flatland-community-logo.svg" alt="Flatland Community"  height="60"/></a>
