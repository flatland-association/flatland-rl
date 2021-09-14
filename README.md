🚂 Flatland
========

![Flatland](https://i.imgur.com/0rnbSLY.gif)

<p style="text-align:center">
<img alt="repository" src="https://gitlab.aicrowd.com/flatland/flatland/badges/master/pipeline.svg">
<img alt="coverage" src="https://gitlab.aicrowd.com/flatland/flatland/badges/master/coverage.svg">
</p>

Flatland is a open-source toolkit for developing and comparing Multi Agent Reinforcement Learning algorithms in little (or ridiculously large!) gridworlds.

[The official documentation](http://flatland.aicrowd.com/) contains full details about the environment and problem statement

Flatland is tested with Python 3.6, 3.7 and 3.8 on modern versions of macOS, Linux and Windows. You may encounter problems with graphical rendering if you use WSL. Your [contribution is welcome](https://flatland.aicrowd.com/misc/contributing.html) if you can help with this!  

🏆 Challenges
---

This library was developed specifically for the AIcrowd [Flatland challenges](http://flatland.aicrowd.com/research/top-challenge-solutions.html) in which we strongly encourage you to take part in!

- [Flatland 3 Challenge](https://www.aicrowd.com/challenges/flatland-3) - ONGOING!
- [AMLD 2021 Challenge](https://www.aicrowd.com/challenges/flatland)
- [NeurIPS 2020 Challenge](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/)
- [2019 Challenge](https://www.aicrowd.com/challenges/flatland-challenge)

📦 Setup
---

### Prerequisites (optional)

Install [Anaconda](https://www.anaconda.com/distribution/) and create a new conda environment:

```console
$ conda create python=3.7 --name flatland-rl
$ conda activate flatland-rl
```

### Stable release

Install Flatland from pip:

```console
$ pip install flatland-rl
```

This is the preferred method to install Flatland, as it will always install the most recent stable release.

### From sources

The Flatland code source is available from [AIcrowd gitlab](https://gitlab.aicrowd.com/flatland/flatland).

Clone the public repository:

```console
$ git clone git@gitlab.aicrowd.com:flatland/flatland.git
```

Once you have a copy of the source, install it with:

```console
$ pip install -e .
```

### Test installation

Test that the installation works:

```console
$ flatland-demo
```

You can also run the full test suite:

```console
python setup.py test
```

👥 Credits
---

This library was developed by [SBB](https://www.sbb.ch/en/), [Deutsche Bahn](https://www.deutschebahn.com/), [SNCF](https://www.sncf.com/en), [AIcrowd](https://www.aicrowd.com/) and [numerous contributors](http://flatland.aicrowd.com/misc/credits.html) and AIcrowd research fellows from the AIcrowd community.

➕ Contributions
---
Please follow the [Contribution Guidelines](https://flatland.aicrowd.com/misc/contributing.html) for more details on how you can successfully contribute to the project. We enthusiastically look forward to your contributions!

💬 Communication
---

* [Discord Channel](https://discord.com/invite/hCR3CZG)
* [Discussion Forum](https://discourse.aicrowd.com/c/neurips-2020-flatland-challenge)
* [Issue Tracker](https://gitlab.aicrowd.com/flatland/flatland/issues/)

🔗 Partners
---

<a href="https://sbb.ch" target="_blank" style="margin-right:30px"><img src="https://annpr2020.ch/wp-content/uploads/2020/06/SBB.png" alt="SBB" width="140"/></a> 
<a href="https://www.deutschebahn.com/" target="_blank" style="margin-right:30px"><img src="https://i.imgur.com/pjTki15.png" alt="DB"  width="140"/></a>
<a href="https://www.sncf.com/en" target="_blank" style="margin-right:30px"><img src="https://iconape.com/wp-content/png_logo_vector/logo-sncf.png" alt="SNCF"  width="140"/></a>
<a href="https://www.aicrowd.com" target="_blank"><img src="https://i.imgur.com/kBZQGI9.png" alt="AIcrowd"  width="140"/></a>
