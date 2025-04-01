# Tsuchinoko

Tsuchinoko is a Qt application for adaptive experiment execution and tuning. Live visualizations show details of
measurements, and provide feedback on the adaptive engine's decision-making process. The parameters of the adaptive
engine can also be tuned live to explore and optimize the search procedure.

```{image} _static/running-score.PNG
:alt: Tsuchinoko with no connection
:align: center

```

&nbsp;

While Tsuchinoko is designed to allow custom adaptive engines to drive experiments, the
[gpCAM](https://gpcam.readthedocs.io/en/latest/) engine is a featured inclusion. This tool is based on a flexible and
powerful Gaussian process regression at the core.

A Tsuchinoko system includes 4 distinct components: the GUI client, an adaptive engine, and execution engine, and a
core service. These components are separable to allow flexibility with a variety of distributed designs.

## Installation

The latest stable Tsuchinoko version is available on PyPI, and is installable with `pip`. It is recommended that you
use Python 3.12 for this installation.

```bash
pip install tsuchinoko
```

For more information, see the [installation documentation](quickstart.md).

## Easy Installation

For Mac OSX and Windows, pre-packaged installers are available. These do not require a base Python installation. See the [installation documentation](https://tsuchinoko.readthedocs.io/en/latest/installers.html) for more details.

- [Latest Windows Installer](https://github.com/lbl-camera/tsuchinoko/releases/latest/download/Tsuchinoko-amd64.exe)
- [Latest Mac OSX Installer](https://github.com/lbl-camera/tsuchinoko/releases/latest/download/Tsuchinoko.app.tgz)

## Getting started with your own adaptive experiments

You can find demos in the Tsuchinoko Github repository's [examples folder](https://github.com/lbl-camera/tsuchinoko/tree/master/examples).
It is suggested to first try running both `server_demo.py` and `client_demo.py` simultaneously. This demo performs a
simulated adaptive experiment by making "measurements" sampled from an image file. It is recommended as a first run to follow
the [Getting Started](quickstart.md) guide.

## About the name

Japanese folklore describes the [Tsuchinoko](https://cryptidz.fandom.com/wiki/Tsuchinoko) as a wide and short snake-like creature living in the mountains of western
Japan. This creature has a cultural following similar to the Bigfoot of North America. Much like the global optimum of a
non-convex function, its elusive nature is infamous.

```{toctree}
---
hidden: true
maxdepth: 1
---
installers.md
quickstart.md
server_experiment.md
bluesky.md
api/index.md
bluesky-adaptive.md
```