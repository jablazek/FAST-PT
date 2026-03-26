.. _theory:

Theory
=====

FAST-PT Algorithm
---------------

FAST-PT is a numerical algorithm to calculate 1-loop contributions to the matter power spectrum and other integrals of a similar type. The method is presented in papers `arXiv:1603.04826 <https://arxiv.org/abs/1603.04826>`_ and `arXiv:1609.05978 <https://arxiv.org/abs/1609.05978>`_.

The core of the FAST-PT algorithm is to compute integrals of the form:

.. math::

   \int \frac{d^3q}{(2 \pi)^3} K(q,k-q) P(q) P(|k-q|)

by using Fourier transforms to reduce computational complexity from :math:`O(N^2)` to :math:`O(N\\log N)`, where :math:`N` is the number of input points.

For intrinsic alignment calculations, see `arXiv:1708.09247 <https://arxiv.org/abs/1708.09247>`_.