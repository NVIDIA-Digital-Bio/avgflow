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

from typing import Optional
from einops import rearrange
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp

from .adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
    Identity
)

def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y

def max_neg_value(x: jnp.ndarray):
    return jnp.finfo(x.dtype).min

class PairBiasAttention(nnx.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    """

    def __init__(
        self, 
        node_dim: int, 
        head_dim: int,
        pair_dim: int, 
        out_dim: int, 
        n_heads: int, 
        bias: bool, 
        qkln: bool, # Whether to layernorm q and k
        rngs: nnx.Rngs,
    ):
        inner_dim = head_dim * n_heads

        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        # Layers to cast input to q, k, v, gate and output
        self.to_qkv = nnx.Linear(node_dim, node_dim * 3, rngs=rngs, use_bias=bias)
        self.to_gate = nnx.Linear(node_dim, inner_dim, rngs=rngs)
        self.to_out = nnx.Linear(inner_dim, default(out_dim, node_dim), rngs=rngs)

        # LayerNorms 
        self.node_ln = nnx.LayerNorm(num_features=node_dim, rngs=rngs) 
        self.q_ln = nnx.LayerNorm(num_features=inner_dim, rngs=rngs) if qkln else Identity()
        self.k_ln = nnx.LayerNorm(num_features=inner_dim, rngs=rngs) if qkln else Identity()

        if exists(pair_dim):
            self.to_bias = nnx.Linear(pair_dim, n_heads, rngs=rngs)
            self.pair_ln = nnx.LayerNorm(num_features=pair_dim, rngs=rngs)
        else:
            raise ValueError("pair_dim is None, this model is designed for data with pairwise input")
    
    def __call__(
        self, 
        node_feats: jnp.array, 
        pair_bias: Optional[jnp.array],
        mask: Optional[jnp.array]
    ) -> jnp.array:
        """Multi-head attention with pair bias and gating.
        node_feats: [b, n, node_dim]
        pair_bias: [b, n, n, pair_feat_dim]
        mask: [b, n, n] Here the mask is the pairwise mask
        """
        assert exists(self.to_bias) or not exists(pair_bias), "pair_bias is not None but pair_dim is None"
        node_feats, n_heads = self.node_ln(node_feats), self.n_heads
        pair_feats = self.pair_ln(pair_bias) if exists(pair_bias) else None

        q, k, v = jnp.split(self.to_qkv(node_feats), 3, axis=-1)
        q = self.q_ln(q)
        k = self.k_ln(k)
        gate = self.to_gate(node_feats)
        bias = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...") if exists(pair_feats) else 0
        )

        q, k, v, gate = map(
            lambda x: rearrange(x, "b ... (h d) -> b h ... d", h=n_heads), (q, k, v, gate)
        )

        attn_feats = self._attn(q, k, v, bias, mask)
        attn_feats = rearrange(
            nnx.sigmoid(gate) * attn_feats, 
            "b h n d -> b n (h d)", h=n_heads
        )
        return self.to_out(attn_feats)

    def _attn(self, q, k, v, bias, mask) -> jnp.array:
        """Compute attention with bias and mask."""
        sim = jnp.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            # sim = jnp.where(~mask, -jnp.inf, sim)
            sim = jnp.where(mask, sim, max_neg_value(sim))
        attn = nnx.softmax(sim + bias, axis=-1)
        return jnp.einsum("b h i j, b h j d -> b h i d", attn, v)


class MultiHeadBiasAttn_ADALN(nnx.Module):
    """
    Multi-head attention with pairwise bias and gating.
    Adaptive layer norm and output scaling is applied to attn output.
    """
    def __init__(
        self, 
        node_dim: int, 
        cond_dim: int,
        pair_dim: int,
        n_heads: int, 
        use_qkln: bool, 
        rngs: nnx.Rngs, 
    ):
        assert node_dim % n_heads == 0, "node_dim must be divisible by n_heads"
        self.head_dim = int(node_dim // n_heads)
        self.adaln = AdaptiveLayerNorm(dim=node_dim, cond_dim=cond_dim, rngs=rngs)

        self.mha = PairBiasAttention(
            node_dim=node_dim, 
            head_dim=self.head_dim,
            pair_dim=pair_dim,
            out_dim=node_dim,
            n_heads=n_heads,
            bias=True,
            qkln=use_qkln,
            rngs=rngs,
        )
        self.scale_output = AdaptiveOutputScale(dim=node_dim, cond_dim=cond_dim, rngs=rngs)

    def __call__(
        self, 
        node_feats: jnp.array, 
        pair_repr: jnp.array,
        cond: jnp.array,
        mask: Optional[jnp.array]
    ) -> jnp.array:
        """
        node_feats: [b, n, node_dim]
        pair_bias: [b, n, n, pair_dim]
        cond: [b, n, cond_dim]
        mask: [b, n]
        """
        pair_mask = mask[:, :, None] * mask[:, None, :] # [b, n, n]
        # print('mask shape:', mask.shape, node_feats.shape, pair_repr.shape, cond.shape)
        node_feats = self.adaln(node_feats, cond, mask)
        node_feats = self.mha(node_feats, pair_repr, pair_mask)
        node_feats = self.scale_output(node_feats, cond, mask)
        # Mask out padding
        # node_feats = jnp.where(mask[..., None], node_feats, jnp.zeros_like(node_feats))
        node_feats = node_feats * mask[..., None]
        return node_feats