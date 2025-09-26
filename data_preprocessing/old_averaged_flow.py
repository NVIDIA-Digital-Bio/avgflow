from functools import partial
from typing import *

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import DiGraphMatcher, GraphMatcher


def ot_sensitivity(t, sigma0=1.0, sigma1=0.0):
    sigma_t = (1 - t) * sigma0 + t * sigma1
    return t / sigma_t**2


def ot_expectation_flow(
    t: jax.Array,  # [] or [num_graphs]
    x: jax.Array,  # [num_nodes, 3]
    x1: jax.Array,  # [num_nodes, (num_conformers,) 3]
    permutations: jax.Array | None = None,  # [num_nodes, num_permutations]
    *,
    perm_mask: jax.Array | None = None,  # [(num_graphs,) num_permutations]
    weights: jax.Array | None = None,  # [(num_graphs,) num_conformers]
    n_node: jax.Array | None = None,  # [num_graphs]
    sigma0: jax.Array = 1.0,
    sigma1: jax.Array = 0.0,
) -> jax.Array:
    r"""
    Compute the flow averaged over the rotations of the input x1 with the bias corresponding to the optimal transport strategy from the Flow Matching paper.

    .. math::

        u_t(x|x_1) = \frac{1}{Z}\int dg \frac{g(x_1) - x}{1 - t}\ e^{-\frac{\| x - t g(x_1) \|^2}{2\sigma_t^2}}


    where p_0 is the Gaussian distribution with variance sigma0^2, and Z is the normalization factor. Z is the same expression as the numerator but without the u factor.

    Args:
        t: the time in the flow matching formalism.
        x: the point at which the flow is evaluated, shape (n, 3).
        x1: the input data points, shape (n, 3) or (n, num_conformers, 3).
        permutations: permutations of x1 to average over
        perm_mask: boolean mask to select which permutations to average over
        weights: weights for each conformer in x1
        n_node: number of nodes in each graph
        sigma0: the variance of :math:`p_t` at time t=0.
        sigma1: the variance of :math:`p_t` at time t=1.
    """

    def sensitivity(t):
        return ot_sensitivity(t, sigma0, sigma1)

    return general_averaged_flow(
        sensitivity, t, x, x1, permutations, perm_mask, weights, n_node
    )


def general_averaged_flow(
    sensitivity: Callable[[jax.Array], jax.Array],
    t: jax.Array,  # [] or [num_graphs]
    x: jax.Array,  # [num_nodes, 3]
    x1: jax.Array,  # [num_nodes, (num_conformers,) 3]
    permutations: jax.Array | None = None,  # [num_nodes, num_permutations]
    perm_mask: jax.Array | None = None,  # [(num_graphs,) num_permutations]
    weights: jax.Array | None = None,  # [(num_graphs,) num_conformers]
    n_node: jax.Array | None = None,  # [num_graphs]
) -> jax.Array:
    t, x, x1, permutations, perm_mask, n_node = jax.tree.map(
        jnp.asarray, (t, x, x1, permutations, perm_mask, n_node)
    )
    num_nodes = x.shape[0]
    num_graphs = n_node.shape[0] if n_node is not None else 1
    num_conformers = x1.shape[1] if x1.ndim == 3 else 1

    if permutations is not None and permutations.ndim != 2:
        raise ValueError(
            f"Expected permutations to have shape (num_nodes, num_permutations), got {permutations.shape}"
        )
    num_permutations = permutations.shape[1] if permutations is not None else 1

    if t.shape not in [(num_graphs,), ()]:
        raise ValueError(
            f"Expected t to have shape ({num_graphs},) or (), got {t.shape}"
        )
    if x.shape != (num_nodes, 3):
        raise ValueError(f"Expected x to have shape ({num_nodes}, 3), got {x.shape}")
    if x1.shape not in [(num_nodes, 3), (num_nodes, num_conformers, 3)]:
        raise ValueError(
            f"Expected x1 to have shape ({num_nodes}, 3) or ({num_nodes}, {num_conformers}, 3), got {x1.shape}"
        )
    if permutations is not None:
        if permutations.shape != (num_nodes, num_permutations):
            raise ValueError(
                f"Expected permutations to have shape ({num_nodes}, {num_permutations}), got {permutations.shape}"
            )

    if t.shape == (num_graphs,):
        sensitivity = jax.vmap(sensitivity)

    if x1.ndim == 2:
        x1 = x1[:, None, :]

    avg_x1 = soft_alignment(sensitivity(t), x1, x, permutations, perm_mask, weights, n_node)

    if n_node is not None and t.shape == n_node.shape:
        t = jnp.repeat(t, n_node, total_repeat_length=x.shape[0])[:, None]

    return (avg_x1 - x) / (1 - t)


@jax.jit
def soft_alignment(
    sensitivity: jax.Array,  # [] or [num_graphs]
    x: jax.Array,  # [num_nodes, num_conformers, 3]
    y: jax.Array,  # [num_nodes, 3]
    permutations: jax.Array | None = None,  # [num_nodes, num_permutations]
    perm_mask: jax.Array | None = None,  # [(num_graphs,) num_permutations]
    weights: jax.Array | None = None,  # [(num_graphs,) num_conformers]
    n_node: jax.Array | None = None,  # [num_graphs]
) -> jax.Array:
    r"""
    Compute biased mean of :math:`g(\sigma(x))` over the rotations :math:`g` in :math:`SO(3)` and the permutations :math:`\sigma` in the provided set.
    The mean is biased towards aligning :math:`g(\sigma(x))` with :math:`y`.

    .. math::

        \langle g(\sigma(x)) \rangle_y = \frac{1}{Z} \sum_x w(x) \sum_{\sigma\in\Sigma} \int dg\ g(\sigma(x)) e^{\beta g(\sigma(x)) \cdot y}

    where :math:`Z` is the normalization factor and :math:`\beta` is the sensitivity parameter.
    :math:`\Sigma` is the set of permutations if provided, otherwise no permutation is applied.
    """
    num_nodes, num_conformers, _ = x.shape
    num_graphs = n_node.shape[0] if n_node is not None else 1
    assert y.shape == (num_nodes, 3)
    assert x.shape == (num_nodes, num_conformers, 3)
    assert sensitivity.shape in [(num_graphs,), ()]

    if n_node is None:
        if sensitivity.shape != ():
            raise ValueError(
                f"n_node is None but sensitivity has shape {sensitivity.shape}, expected ()"
            )

        def logZ(alpha):
            def f(x_):
                return logcF(sensitivity * x_.T @ y + x_.T @ alpha)

            x_ = jnp.moveaxis(x, 1, 0)  # [num_conformers, num_nodes, 3]

            if permutations is not None:
                x_ = x_[
                    :, permutations.T, :
                ]  # [num_conformers, num_permutations, num_nodes, 3]
                a = jax.vmap(jax.vmap(f))(x_)  # [num_conformers, num_permutations]
                a = jax.vmap(partial(logsumexp, weights=perm_mask))(a)
            else:
                a = jax.vmap(f)(x_)  # [num_conformers]

            return logsumexp(a, weights)

        return jax.grad(logZ)(jnp.zeros_like(y))

    else:
        if sensitivity.shape == ():
            sensitivity = jnp.repeat(sensitivity, num_graphs)

        def logZ(alpha):
            def f(x_):
                assert x_.shape == (num_nodes, 3)
                xy = jnp.einsum("ni,nj->nij", x_, y)
                xa = jnp.einsum("ni,nj->nij", x_, alpha)

                xy = partition_sum(xy, n_node)  # [num_graphs, 3, 3]
                xa = partition_sum(xa, n_node)  # [num_graphs, 3, 3]

                F = sensitivity[:, None, None] * xy + xa  # [num_graphs, 3, 3]
                return jax.vmap(logcF)(F)  # [num_graphs]

            x_ = jnp.moveaxis(x, 1, 0)  # [num_conformers, num_nodes, 3]

            if permutations is not None:
                node_offset = jnp.repeat(
                    jnp.append(0, jnp.cumsum(n_node[:-1])),
                    n_node,
                    total_repeat_length=num_nodes,
                )
                i = permutations + node_offset[:, None]
                x_ = x_[:, i.T, :]  # [num_conformers, num_permutations, num_nodes, 3]
                a = jax.vmap(jax.vmap(f))(x_)
                # ^^ [num_conformers, num_permutations, num_graphs]
                a = jax.vmap(lambda a_: jax.vmap(logsumexp, (1, 0))(a_, perm_mask))(a)
                # ^^ [num_conformers, num_graphs]
            else:
                a = jax.vmap(f)(x_)  # [num_conformers, num_graphs]

            a = jax.vmap(logsumexp, (1, 0))(a, weights)  # [num_graphs]
            return jnp.sum(a)

        return jax.grad(logZ)(jnp.zeros_like(y))


def logsumexp(a: jax.Array, weights: jax.Array | None = None) -> jax.Array:
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


@jax.jit
def partition_sum(data: jax.Array, partition: jax.Array) -> jax.Array:
    segment_ids = jnp.repeat(
        jnp.arange(partition.shape[0]), partition, total_repeat_length=data.shape[0]
    )
    return jax.ops.segment_sum(
        data,
        segment_ids,
        partition.shape[0],
        indices_are_sorted=True,
        unique_indices=False,
        bucket_size=None,
    )


# This content below is a Pytorch to JAX port of the code from the following repository from David Mohlin, Gérald Bianchi and Josephine Sullivan
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


def bessel0(x: jax.Array) -> jax.Array:
    p = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
    bessel0_a = jnp.array(p[::-1])

    p = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2]
    p += [-0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]
    bessel0_b = jnp.array(p[::-1])

    abs_x = jnp.abs(x)
    x_lim = 3.75

    def w(x, y):
        return jnp.where(abs_x <= x_lim, x, y)

    abs_x_ = w(x_lim, abs_x)

    return w(
        jnp.polyval(bessel0_a, w(abs_x / x_lim, 1.0) ** 2) * jnp.exp(-abs_x),
        jnp.polyval(bessel0_b, w(1.0, x_lim / abs_x_)) / jnp.sqrt(abs_x_),
    )


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
    x = jnp.linspace(0.0, 1.0, 512)
    r2 = jnp.trapezoid(jax.vmap(f)(x) * jnp.exp(-a * x), x)

    return jnp.where(a > 1.0, r1, r2)


@jax.custom_vjp
def logcf(s1: jax.Array, s2: jax.Array, s3: jax.Array) -> jax.Array:
    # assume s1 >= s2 >= s3
    s1, s2, s3 = jnp.asarray(s1), jnp.asarray(s2), jnp.asarray(s3)
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


def signed_svdvals(F: jax.Array) -> jax.Array:
    u, s, vh = jnp.linalg.svd(F, full_matrices=False)
    u, vh = jax.lax.stop_gradient((u, vh))
    sign = jnp.sign(jnp.linalg.det(u @ vh))
    return s.at[-1].mul(sign)


# Simple implementation of Kabsch algorithm and permutation search


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
    q = vh.T @ jnp.diag(jnp.array([1, 1, d])) @ u.T
    return q


@jax.jit
def kabsch_with_permutation(
    x: jax.Array, y: jax.Array, permutations: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Find rotation matrix R and permutation p that minimizes the RMSD between x[p] @ R.T and y."""
    # ZC: RMSD = norm((x-y)/n_atoms)
    def fn(i):
        R = kabsch(x[i], y)
        err = jnp.linalg.norm((x[i] @ R.T - y))/jnp.sqrt(x.shape[0])
        return R, err

    Rs, errs = jax.vmap(fn)(permutations)
    best = jnp.argmin(errs)
    return Rs[best], permutations[best], errs[best]



@jax.jit
def kabsch_with_permutation_and_inversion(
    x: jax.Array, y: jax.Array, permutations: jax.Array
) -> tuple[jax.Array, jax.Array]:
    Rs, perm, err = kabsch_with_permutation(x, y, permutations)
    Rs2, perm2, err2 = kabsch_with_permutation(-x, y, permutations)
    first = err<err2
    return jnp.where(first, Rs, Rs2), jnp.where(first, perm, perm2), ~first, jnp.where(first, err, err2)


##### Identifying the permutation of a graph #####


def graph_permutations(
        node_features: np.ndarray,
        senders: np.ndarray,
        receivers: np.ndarray,
        edge_features: np.ndarray,
        max_permutations: int = None,
        bidirectional: bool = False,
    ) -> np.ndarray:
    G = nx.DiGraph() if bidirectional else nx.Graph()

    for i, f in enumerate(node_features):
        G.add_node(i, feature=f)
    for s, r, f in zip(senders, receivers, edge_features):
        G.add_edge(s, r, feature=f)

    GM = (DiGraphMatcher if bidirectional else GraphMatcher)(
        G,
        G,
        node_match=lambda n1, n2: np.array_equal(n1["feature"], n2["feature"]),
        edge_match=lambda e1, e2: np.array_equal(e1["feature"], e2["feature"]),
    )

    permutations = []

    identity = list(G.nodes)
    permutations.append(identity)
    for iso in GM.isomorphisms_iter():
        perm = [iso[n] for n in G.nodes]
        if perm == identity:
            continue

        permutations.append(perm)
        if max_permutations is not None and len(permutations) >= max_permutations:
            break

    permutations = np.array(permutations)
    return permutations


def graph_permutations_Hmask(
        node_features: np.ndarray,
        senders: np.ndarray,
        receivers: np.ndarray,
        edge_features: np.ndarray,
        max_permutations: int = None,
        bidirectional: bool = False,
        dataset = 'drugs',
    ) -> np.ndarray:

    G = nx.DiGraph() if bidirectional else nx.Graph()

    if dataset == 'drugs':
        node_features = node_features[:, :35]
    elif dataset == 'qm9':
        node_features = node_features[:, :5]

    for i, f in enumerate(node_features):
        if f[0] == 1:
            f = np.zeros_like(f)
            f[0] = 2024+i
        G.add_node(i, feature=f)
    for s, r, f in zip(senders, receivers, edge_features):
        G.add_edge(s, r, feature=f)

    GM = (DiGraphMatcher if bidirectional else GraphMatcher)(
        G,
        G,
        node_match=lambda n1, n2: np.array_equal(n1["feature"], n2["feature"]),
        edge_match=lambda e1, e2: np.array_equal(e1["feature"], e2["feature"]),
    )
    saved = set()
    permutations = []

    identity = list(G.nodes)
    permutations.append(identity)
    for iso in GM.isomorphisms_iter():
        perm = [iso[n] for n in G.nodes]
        if perm == identity:
            continue

        permutations.append(perm)
        saved.add(','.join(map(str, perm)))
        if max_permutations is not None and len(permutations) >= max_permutations:
            break

    ##### hydrogen permutations #####
    G = nx.DiGraph() if bidirectional else nx.Graph()

    if dataset == 'drugs':
        node_features = node_features[:, :35]
    elif dataset == 'qm9':
        node_features = node_features[:, :5]

    for i, f in enumerate(node_features):
        G.add_node(i, feature=f)
    for s, r, f in zip(senders, receivers, edge_features):
        G.add_edge(s, r, feature=f)

    GM = (DiGraphMatcher if bidirectional else GraphMatcher)(
        G,
        G,
        node_match=lambda n1, n2: np.array_equal(n1["feature"], n2["feature"]),
        edge_match=lambda e1, e2: np.array_equal(e1["feature"], e2["feature"]),
    )
    
    for iso in GM.isomorphisms_iter():
        perm = [iso[n] for n in G.nodes]
        if perm == identity:
            continue

        key = ','.join(map(str, perm))
        if key in saved:
            continue
        # Prioritize non-hydrogen permutations
        permutations.append(perm)
        if max_permutations is not None and len(permutations) >= max_permutations:
            break
    permutations = np.array(permutations[:max_permutations])
    return permutations