# FAST-PT

[![Documentation Status](https://readthedocs.org/projects/fast-pt/badge/?version=latest)](https://fast-pt.readthedocs.io/en/latest/?badge=latest)
[![arXiv:1603.04826](https://img.shields.io/badge/arXiv-1603.04826-b31b1b.svg)](https://arxiv.org/abs/1603.04826)
[![arXiv:1609.05978](https://img.shields.io/badge/arXiv-1609.05978-b31b1b.svg)](https://arxiv.org/abs/1609.05978)
[![arXiv:1708.09247](https://img.shields.io/badge/arXiv-1708.09247-b31b1b.svg)](https://arxiv.org/abs/1708.09247)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FAST-PT is a code to calculate quantities in cosmological perturbation theory
at 1-loop (including, e.g., corrections to the matter power spectrum). The code
utilizes Fourier methods combined with analytic expressions to reduce the
computation time to scale as N log N, where N is the number of grid points in
the input linear power spectrum.

NOTE: v3.1.0 and earlier require numpy version < 1.24. This is fixed in v3.1.1 and later, which is available on pip and conda.

Easy installation with pip:

* `pip install fast-pt`
* Note: use `--no-deps` if you use a conda python distribution, or just use conda installation

Easy installation with conda:

* `conda install fast-pt`

Full installation with examples:

* Make sure you have current version of numpy, scipy, and matplotlib
* download the latest FAST-PT release (or clone the repo)
* install the repo: `python -m pip install .`
* run the most recent example: `cd examples && python3 hello_fastpt.py`
* hopefully you get a plot!
* for a more in-depth example of new features:  `cd examples && python3 v4_example.py`
* for older examples see the 'examples' folder

Our papers (JCAP 2016, 9, 15; arXiv:1603.04826) and (JCAP 2017, 2, 30; arXiv:1609.05978)
describe the FAST-PT algorithm and implementation. Please cite these papers
when using FAST-PT in your research. For the intrinsic alignment
implementation, cite PRD 100, 103506 (arXiv:1708.09247).

FAST-PT is under continued development and should be considered research in
progress. FAST-PT is open source and distributed with the
[MIT license](https://opensource.org/licenses/mit). If you have comments,
questions, or feedback, please file an [issue](https://github.com/jablazek/FAST-PT/issues).
