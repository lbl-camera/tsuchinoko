# Tsuchinoko
[![PyPI](https://badgen.net/pypi/v/tsuchinoko)](https://pypi.org/project/tsuchinoko/)
[![License](https://badgen.net/pypi/license/tsuchinoko)](https://github.com/ronpandolfi/tsuchinoko)
[![Build Status](https://img.shields.io/travis/Xi-cam/Xi-cam/master.svg)](https://travis-ci.org/Xi-cam/Xi-cam)
[![Documentation Status](https://readthedocs.org/projects/tsuchinoko/badge/?version=latest)](https://tsuchinoko.readthedocs.io/en/latest/?badge=latest)
[![Test Coverage](https://img.shields.io/codecov/c/github/ronpandolfi/tsuchinoko/master.svg)](https://codecov.io/github/ronpandolfi/tsuchinoko?branch=master)
[![Slack Status](https://img.shields.io/badge/slack-gpCAM-yellow.svg?logo=slack)](https://nikea.slack.com/messages/U7Q1N42F6)

Tsuchinoko is a Qt application for adaptive experiment execution and tuning. Live visualizations show details of 
measurements, and provide feedback on the adaptive engine's decision-making process. The parameters of the adaptive 
engine can also be tuned live to explore and optimize the search procedure.

While Tsuchinoko is designed to allow custom adaptive engines to drive experiments, the 
[gpCAM](https://gpcam.readthedocs.io/en/latest/) engine is a featured inclusion. This tool is based on a flexible and 
powerful Gaussian process regression at the core.

A Tsuchinoko system includes 4 distinct components: the GUI client, an adaptive engine, and execution engine, and a
core service. These components are separable to allow flexibility with a variety of distributed designs.

## Installation

The latest stable Tsuchinoko version is available on PyPI, and is installable with `pip`.

```bash
pip install tsuchinoko
```

For more information, see the [installation documentation](https://xi-cam.readthedocs.io/en/latest/quickstart.html).

## Resources

* Documentation: https://tsuchinoko.readthedocs.io/en/latest
* Report an issue in Xi-CAM: [New Bug Report](https://github.com/ronpandolfi/tsuchinoko/issues/new?labels=bug)

## About the name

Japanese folklore describes the [Tsuchinoko](https://cryptidz.fandom.com/wiki/Tsuchinoko) as a wide and short snake-like creature living in the mountains of western
Japan. This creature has a following much like the Bigfoot of North America. Much like the global optimum of a
non-convex function, its elusive nature is infamous.