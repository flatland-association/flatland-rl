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

This is the preferred method to install Flatland, as it will always install the most recent stable release. The
published package is pure Python, so this always works regardless of your platform or toolchain.

### Cython-accelerated build

A few hot-path modules can optionally be compiled with [Cython](https://cython.org/) for extra performance. This requires building from
source, since the package published on PyPI is the plain-Python build described above:

```shell
python -m pip install "cython>=3.2.9"
python -m pip install --no-build-isolation --no-binary flatland-rl flatland-rl
```

(or `python -m pip install --no-build-isolation -e .` from a checkout of this repository).

**Why `--no-binary` and `--no-build-isolation`?**

- `--no-binary flatland-rl` forces pip to build from the source distribution (sdist) instead of installing the
  prebuilt wheel (bdist) already published on PyPI. This matters because that published wheel is a *universal*
  one - e.g. `flatland_rl-4.2.6-py2.py3-none-any.whl` (`none-any`: no Python-version/ABI/platform constraint) -
  which pip considers compatible with any environment, so by default pip always prefers installing it over
  building from the `flatland_rl-4.2.6.tar.gz` sdist. Without `--no-binary`, pip just installs that existing
  plain-Python wheel and never runs a build at all, so there's nothing to cythonize, regardless of whether
  Cython or a C compiler is available locally.
- `--no-build-isolation` is needed because Cython is *not* listed in `pyproject.toml`'s `[build-system]
  requires` (only `setuptools`/`setuptools_scm` are - deliberately, so a normal `pip install flatland-rl` isn't
  forced to pull in Cython just to install an already-published wheel). By default pip builds every package
  inside a fresh, throwaway virtualenv containing only those `[build-system] requires` packages, so even a
  Cython you'd `pip install`ed into your own environment would be invisible to that isolated build. This flag
  tells pip to skip the throwaway environment and build using your current environment's packages instead, so
  the Cython you installed a line above actually gets picked up.

**Requirements:**

- [Cython](https://cython.org/) `>=3.2.9` installed in the environment you are installing into (`pip install
  cython` above) - it must be present *before* pip starts the build, since `--no-build-isolation` (see above)
  is what makes it visible to the build in the first place.
- A working C compiler (e.g. `gcc`/`clang` on Linux/macOS, or the Microsoft C++ Build Tools on Windows).

Cython compilation is optional and best-effort: if either requirement is missing, the build falls back to the
plain-Python sources instead of failing, printing a warning for each module it could not compile. Pass `-v` to
pip (`python -m pip install -v --no-build-isolation ...`) to see it - by default pip hides the underlying build
output on success.

In both cases the install still succeeds - you get the plain-Python modules, just without the Cython speed-up.

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
