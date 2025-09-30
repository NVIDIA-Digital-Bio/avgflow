# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import jax
import jax.numpy as jnp


def kabsch(x: jax.Array, y: jax.Array) -> jax.Array:
    """Find rotation matrix R that minimizes the RMSD between x @ R.T and y."""
    n = x.shape[0]
    assert x.shape == (n, 3)
    assert y.shape == (n, 3)
    x = x - jnp.mean(x, axis=0)
    y = y - jnp.mean(y, axis=0)
    h = x.T @ y
    u, _, vh = jnp.linalg.svd(h)
    d = jnp.sign(jnp.linalg.det(vh.T @ u.T))
    R = vh.T @ jnp.diag(jnp.array([1, 1, d])) @ u.T
    return R


def sample_exp_neg_half_xAx(key: jax.Array, A: jax.Array) -> jax.Array:
    """Sample a random vector x from the distribution exp(-0.5 * x^T A x)."""
    n = A.shape[0]
    w, V = jnp.linalg.eigh(A)
    z = jax.random.normal(key, shape=(n,), dtype=A.dtype)
    return V @ (z / jnp.sqrt(w))


def avg_harmonic_flow(
    t: jax.Array,  # []
    x: jax.Array,  # [num_nodes, 3]
    x1: jax.Array,  # [num_conformers, num_nodes, 3]
    edges: jax.Array,  # [2, num_edges]
    target_weights: jax.Array | None = None,  # [num_conformers]
    edge_weights: jax.Array | None = None,  # [num_edges]
    sigma0: jax.Array = 1.0,
    sigma1: jax.Array = 0.0,
) -> jax.Array:
    # edges: example [[0, 1], [0, 2], [1, 2]].T
    if edge_weights is None:
        edge_weights = jnp.ones_like(edges[0], dtype=x.dtype)

    degree = jnp.bincount(edges[0], length=x.shape[0], weights=edge_weights)
    degree += jnp.bincount(edges[1], length=x.shape[0], weights=edge_weights)

    def metric(x, y):  # [num_nodes], [num_nodes] -> []
        laplacian = (
            jnp.sum(degree * x * y)
            - jnp.sum(x[edges[0]] * y[edges[1]] * edge_weights)
            - jnp.sum(x[edges[1]] * y[edges[0]] * edge_weights)
        )
        return laplacian / ((1 - t) * sigma0 + t * sigma1) ** 2

    avg_x1 = avg_target(x, x1, t, metric, target_weights)

    return (avg_x1 - x) / (1 - t)


def avg_flow(
    t: jax.Array,  # []
    x: jax.Array,  # [num_nodes, 3]
    x1: jax.Array,  # [num_conformers, num_nodes, 3]
    target_weights: jax.Array | None = None,  # [num_conformers]
    node_weights: jax.Array | None = None,  # [num_nodes]
    sigma0: jax.Array = 1.0,
    sigma1: jax.Array = 0.0,
) -> jax.Array:
    r"""
    Compute the optimal transport average flow evaluated at x at time t.

    Args:
        t: the time in the flow. Between 0 and 1.
        x: an array of shape (num_nodes, 3). Often denoted as x_t.
        x1: the target conformers. Shape (num_conformers, num_nodes, 3).
        weights: the weights of the conformers. Shape (num_conformers).
        sigma0: the initial variance of the Gaussian kernel. Default is 1.0.
        sigma1: the final variance of the Gaussian kernel. Default is 0.0.

    Returns:
        The flow at time t evaluated at x. An array of shape (num_nodes, 3).

    Note:
        Benchmark on CPU (Apple M2)
        With num_nodes=1000 (the number of nodes is not affecting much the time)

        +----------------+-------+
        | num_conformers | time  |
        +================+=======+
        | 1              | 200μs |
        +----------------+-------+
        | 10             | 500μs |
        +----------------+-------+
        | 100            | 2.3ms |
        +----------------+-------+
        | 1000           | 14ms  |
        +----------------+-------+
    """
    if node_weights is None:
        node_weights = 1.0

    def metric(x, y):
        # x and y have shape [num_nodes]
        sigma_t = (1 - t) * sigma0 + t * sigma1
        return jnp.sum(x * y * node_weights) / sigma_t**2

    avg_x1 = avg_target(x, x1, t, metric, target_weights)

    return (avg_x1 - x) / (1 - t)


def avg_target(
    x: jax.Array,  # [n, 3]
    targets: jax.Array,  # [num_targets, n, 3]
    t: jax.Array,  # []
    metric: Callable[[jax.Array, jax.Array], jax.Array],
    target_weights: jax.Array | None = None,  # [num_targets]
) -> jax.Array:
    r"""
    Among all the targets and all their rotations around the origin,
    compute the average of the rotated targets that are closest to x according to the metric.

    The largest is the metric, the closest to the closest rotated target the result will be.

    .. math::

        \frac{1}{Z} \sum_{\hat x \in targets} weights(\hat x) \int_SO3 dR R(\hat x) e^{-\frac12 \|t R(\hat x) - x\|^2}

    where :math:`Z` is the normalization factor.

    Args:
        x: an array of shape (n, 3).
        targets: an array of shape (num_targets, n, 3).
        t: the time in the flow. Between 0 and 1.
        metric: a function to compute the scalar product of two (n,) vectors. Use to compute the above norm.
        weights: an array of shape (num_targets,).

    Returns:
        an array of shape (n, 3). The average of the rotated targets that are closest to x.
    """
    num_targets, n, _ = targets.shape
    assert x.shape == (n, 3)
    assert targets.shape == (num_targets, n, 3), targets.shape

    def outer(u, v):  # [n, 3] x [n, 3] -> [3, 3]
        return jax.vmap(jax.vmap(metric, (None, -1)), (-1, None))(u, v)

    def inner(u, v):  # [n, 3] x [n, 3] -> []
        return jnp.sum(jax.vmap(metric, (-1, -1))(u, v))

    def logZ(alpha):  # [n, 3] -> []
        def f(target):  # [n, 3] -> []
            return (
                logcF(t * outer(target, x) + target.T @ alpha)
                - (inner(x, x) + t**2 * inner(target, target)) / 2
            )

        return logsumexp(jax.vmap(f)(targets), target_weights)

    return jax.grad(logZ)(jnp.zeros_like(x))


def logsumexp(a: jax.Array, weights: jax.Array | None = None) -> jax.Array:
    # There was a bug in jax.nn.logsumexp but it might have been fixed since then
    # return jax.nn.logsumexp(a, axis=0, keepdims=False, b=weights)

    assert a.ndim == 1
    assert weights is None or weights.shape == a.shape
    where = (weights > 0) if weights is not None else None

    amax = jnp.max(a, where=where, initial=-jnp.inf)
    amax = jax.lax.stop_gradient(
        jax.lax.select(jnp.isfinite(amax), amax, jax.lax.full_like(amax, 0))
    )
    if where is not None:
        a = jnp.where(where, a, amax)
    exp_a = jax.lax.exp(jax.lax.sub(a, amax))
    if weights is not None:
        exp_a = exp_a * weights
    sumexp = exp_a.sum(where=where)
    return jax.lax.add(jax.lax.log(sumexp), amax)


# This content below is adapted from a code by David Mohlin, Gérald Bianchi and Josephine Sullivan
# https://proceedings.neurips.cc/paper_files/paper/2020/file/33cc2b872dfe481abef0f61af181dfcf-Paper.pdf
# https://github.com/Davmo049/Public_prob_orientation_estimation_with_matrix_fisher_distributions/blob/master/torch_norm_factor.py#L66-L90


def logcF(F: jax.Array) -> jax.Array:
    r"""
    Compute

    .. math::

        \log \int_{SO(3)} \exp(\text{tr}(F^T R)) dR
    """
    assert F.shape == (3, 3)
    return logcf(*signed_svdvals(F))


def signed_svdvals(F: jax.Array) -> jax.Array:
    u, s, vh = jnp.linalg.svd(F, full_matrices=False)
    u, vh = jax.lax.stop_gradient((u, vh))
    sign = jnp.sign(jnp.linalg.det(u @ vh))
    return s.at[-1].mul(sign)


@jax.custom_vjp
def logcf(s1: jax.Array, s2: jax.Array, s3: jax.Array) -> jax.Array:
    # assume s1 >= s2 >= s3
    s1, s2, s3 = jax.tree.map(jnp.asarray, (s1, s2, s3))
    return s1 + s2 + s3 + jnp.log(factor(False, s1, s2, s3))


def _logcf_fwd(
    s1: jax.Array, s2: jax.Array, s3: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    # s1 >= s2 >= s3
    f = factor(False, s1, s2, s3)
    return s1 + s2 + s3 + jnp.log(f), (s1, s2, s3, f)


def _logcf_bwd(res: tuple[jax.Array, ...], grad: jax.Array) -> tuple[jax.Array]:
    s1, s2, s3, f = res
    # s1 >= s2 >= s3
    assert s1.shape == ()
    assert f.shape == ()
    assert grad.shape == ()
    g1 = grad * factor(True, s1, s2, s3) / f
    g2 = grad * factor(True, s2, s1, s3) / f
    g3 = grad * factor(True, s3, s1, s2) / f
    return g1, g2, g3


logcf.defvjp(_logcf_fwd, _logcf_bwd)


def factor(add_x: bool, s1: jax.Array, s2: jax.Array, s3: jax.Array) -> jax.Array:
    def f(x):
        i0 = (1.0 - 2 * x) if add_x else 1.0
        i1 = bessel0((s2 - s3) * x)
        i2 = bessel0((s2 + s3) * (1 - x))
        return i0 * i1 * i2

    tiny = jnp.finfo(s1.dtype).tiny
    a = 2 * (s3 + s1)

    # a non zero:
    a_ = jnp.maximum(a, 0.5)
    y = jnp.linspace(tiny + jnp.exp(-a_), 1.0, 512)
    r1 = jnp.trapezoid(jax.vmap(f)(-jnp.log(y) / a_), y) / a_

    # a (close to) zero:
    x = jnp.linspace(0.0, 1.0, 512, dtype=s1.dtype)
    r2 = jnp.trapezoid(jax.vmap(f)(x) * jnp.exp(-a * x), x)

    return jnp.where(a > 1.0, r1, r2)


def bessel0(x: jax.Array) -> jax.Array:
    p = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
    bessel0_a = jnp.array(p[::-1], dtype=x.dtype)

    p = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2]
    p += [-0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
    bessel0_b = jnp.array(p[::-1], dtype=x.dtype)

    abs_x = jnp.abs(x)
    x_lim = 3.75

    def w(x, y):
        return jnp.where(abs_x <= x_lim, x, y)

    abs_x_ = w(x_lim, abs_x)

    return w(
        jnp.polyval(bessel0_a, w(abs_x / x_lim, 1.0) ** 2) * jnp.exp(-abs_x),
        jnp.polyval(bessel0_b, w(1.0, x_lim / abs_x_)) / jnp.sqrt(abs_x_),
    )
