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

import math
from typing import Any, Tuple
from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from .adaptive_ln_scale import AdaptiveLayerNorm, Identity

def get_time_embedding(t, edim, max_positions=2000):
    """
    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    assert edim % 2 == 0 # "Only support even dimension for now"
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if edim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, ((0, 0), (0, 1)), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb

# Atomic representations
"""
Input to the atomic representation include:
    - xt: coordinates of atoms at time t (shape: [b, n, 3])
    - xt_hat: self-conditioning coordinates of atoms at time t (shape: [b, n, 3])
    - atomic features: atomic features (shape: [b, n, n_features])
    - laplacian positional encoding: (shape: [b, n, laplacian_dim]) default dim: 32
    - mask: mask for atoms (shape: [b, n])
"""
class AtomicRepresentation(nnx.Module):
    def __init__(self, max_n: int, input_dim: int, output_dim: int, rngs: nnx.Rngs, use_layer_norm: bool = False):
        key = rngs.params()
        self.max_n = max_n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_norm = nnx.LayerNorm(num_features=input_dim, rngs=rngs) if use_layer_norm else Identity()
        self.linear_out = nnx.Linear(in_features=input_dim, out_features=output_dim, use_bias=False, rngs=rngs)

    def __call__(self, xt, xt_hat, atomic_features, laplacian_pos_enc, mask):
        """
        xt: [b, n, 3]
        atomic_features: [b, n, n_features]
        laplacian_pos_enc: [b, n, laplacian_dim]
        mask: [b, n]
        """
        b, n, _ = xt.shape
        x = jnp.concatenate([xt, xt_hat, atomic_features, laplacian_pos_enc], axis=-1)
        x = self.linear_out(self.layer_norm(x))
        x = jnp.where(mask[..., None], x, jnp.zeros_like(x))
        return x


class ConditionBlock(nnx.Module):
    def __init__(
        self, 
        max_n: int, 
        graph_feat_dim: int,
        cond_dim: int, 
        rngs: nnx.Rngs, 
        use_layer_norm: bool = False
    ):
        key = rngs.params()
        self.max_n = max_n
        self.cond_dim = cond_dim
        self.layer_norm = nnx.LayerNorm(num_features=cond_dim*2, rngs=rngs) if use_layer_norm else Identity()
        self.global_linear_proj = nnx.Linear(in_features=graph_feat_dim, out_features=cond_dim, use_bias=False, rngs=rngs)
        self.linear_out = nnx.Linear(in_features=cond_dim*2, out_features=cond_dim, use_bias=False, rngs=rngs)

    def __call__(self, time, graph_feats, mask):
        """
        time: [b]
        graph_feats: [b, graph_feat_dim]
        mask: [b, n]
        """
        # Get time embeddings, here dimesion is [b, input_dim]
        time_emb = get_time_embedding(time, self.cond_dim)
        # Reshape the time embeddings to [b, 1, input_dim] and repeat it n times to get [b, n, input_dim]
        time_emb = time_emb[:, None, :].repeat(self.max_n, axis=1)
        # Project the global features to the input dimension, and repeat it n times to get [b, n, input_dim]
        graph_feats = self.global_linear_proj(graph_feats[:, None, :])
        graph_feats = graph_feats.repeat(self.max_n, axis=1)
        # Concatenate the time embeddings and global features
        cond = jnp.concatenate([time_emb, graph_feats], axis=-1)
        # Mask the condition based on padding and project it to the output dimension
        cond = jnp.where(mask[..., None], cond, jnp.zeros_like(cond))
        cond = self.linear_out(self.layer_norm(cond))
        # Mask the output based on padding
        cond = jnp.where(mask[..., None], cond, jnp.zeros_like(cond))
        return cond  


class PairwiseRepresentation(nnx.Module):
    def __init__(self, n_edge_type: int, edge_type_emb_dim: int, rngs: nnx.Rngs):
        key = rngs.params()
        self.input_dim = n_edge_type
        self.output_dim = edge_type_emb_dim
        self.edge_emb = nnx.Embed(num_embeddings=n_edge_type, features=edge_type_emb_dim, rngs=rngs)

    def __call__(self, xt, xt_hat, senders, receivers, edge_type, pair_mask):
        """
        xt: [b, n, 3]
        xt_hat: [b, n, 3]
        senders: [b, ne]
        receivers: [b, ne]
        edge_type: [b, ne]
        pair_mask: [b, n, n]
        """
        b, n, _ = xt.shape
        # Get the edge embeddings
        edge_emb = self.edge_emb(edge_type)
        # Initialize the pairwise edge matrix
        pairwise_edge_matrix = jnp.zeros((b, n, n, self.output_dim))
        # Fill the pairwise edge matrix with the edge embeddings
        pairwise_edge_matrix = pairwise_edge_matrix.at[jnp.arange(b)[:, None], senders, receivers].set(edge_emb)
        pairwise_edge_matrix = pairwise_edge_matrix.at[jnp.arange(b)[:, None], receivers, senders].set(edge_emb)
        # Construct the pairwise distance matrix
        xt_pairwise_dist_matrix = jnp.linalg.norm(xt[:, :, None, :] - xt[:, None, :, :], axis=-1)
        xt_hat_pairwise_dist_matrix = jnp.linalg.norm(xt_hat[:, :, None, :] - xt_hat[:, None, :, :], axis=-1)
        # Concatenate the pairwise edge matrix with the pairwise distance matrix
        x = jnp.concatenate(
            [pairwise_edge_matrix, xt_pairwise_dist_matrix[..., None], xt_hat_pairwise_dist_matrix[..., None]], 
            axis=-1
        )
        # Mask the output based on padding
        x = jnp.where(pair_mask[..., None], x, jnp.zeros_like(x))
        return x

# TODO: finish this
class PairwiseBlock(nnx.Module):
    def __init__(self, n_edge_type: int, edge_type_emb_dim: int, cond_dim: int, rngs: nnx.Rngs):
        key = rngs.params()
        self.n_edge_type = n_edge_type
        self.edge_type_emb_dim = edge_type_emb_dim
        self.dim = edge_type_emb_dim + 2 # edge_type_emb_dim + 2 for xt and xt_hat pairwise distances
        self.cond_dim = cond_dim
        self.pair_repr_layer = PairwiseRepresentation(n_edge_type, edge_type_emb_dim, rngs)
        self.adaln = AdaptiveLayerNorm(dim=self.dim, cond_dim=self.cond_dim, rngs=rngs)
        self.cond_transform = nnx.Sequential(
            nnx.LayerNorm(num_features=self.cond_dim, rngs=rngs),
            nnx.Linear(self.cond_dim, self.cond_dim, use_bias=False, rngs=rngs),
        )

    def __call__(self, xt, xt_hat, senders, receivers, edge_type, time, mask):
        """
        xt: [b, n, 3]
        xt_hat: [b, n, 3]
        senders: [b, ne]
        receivers: [b, ne]
        edge_type: [b, ne]
        time: [b]
        mask: [b, n]
        """
        b, n, _ = xt.shape
        pair_mask = mask[:, :, None] * mask[:, None, :]
        pair_repr = self.pair_repr_layer(xt, xt_hat, senders, receivers, edge_type, pair_mask)
        # Get time embeddings, here dimesion is [b, cond_dim]
        time_emb = get_time_embedding(time, self.cond_dim)
        # Reshape the time embeddings to [b, 1, cond_dim] and repeat it n times to get [b, n, cond_dim]
        time_emb = time_emb[:, None, None, :].repeat(n, axis=1).repeat(n, axis=2)
        # Transform the condition to the required dimension
        cond = self.cond_transform(time_emb)
        # Apply the adaptive layer norm
        x = self.adaln(pair_repr, cond, pair_mask)
        return x
