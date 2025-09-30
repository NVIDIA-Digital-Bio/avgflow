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
from .pair_bias_attn import PairBiasAttention, MultiHeadBiasAttn_ADALN

class SwiGLU(nnx.Module):
    """SwiGLU layer"""

    def __call__(self, x):
        x, gates = jnp.split(x, 2, axis=-1)
        return nnx.silu(gates) * x

class Transition(nnx.Module):
    """Transition layer"""
    def __init__(self, dim: int, rngs: nnx.Rngs, expension_factor: int = 4, layer_norm: bool = False):
        inner_dim = int(dim * expension_factor)
        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.ln = nnx.LayerNorm(num_features=dim, rngs=rngs)
        else:
            self.ln = Identity()
        self.swish_linear = nnx.Sequential(
            nnx.Linear(in_features=dim, out_features=inner_dim*2, rngs=rngs, use_bias=False),
            SwiGLU()
        )
        self.linear_out = nnx.Linear(in_features=inner_dim, out_features=dim, rngs=rngs, use_bias=False)
    
    def __call__(self, x, mask):
        """
        x: input representation of shape [b, n, dim]
        mask: binary mask, shape [b, n]
        """
        x = self.ln(x)
        x = self.swish_linear(x)
        x = self.linear_out(x)
        return x * mask[..., None]

class TransitionADALN(nnx.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""
    def __init__(
        self, 
        dim: int, 
        cond_dim: int, 
        rngs: nnx.Rngs, 
        expension_factor: int=4
    ):
        self.adaln = AdaptiveLayerNorm(dim=dim, cond_dim=cond_dim, rngs=rngs)
        self.transition = Transition(dim=dim, rngs=rngs, expension_factor=expension_factor, layer_norm=False)
        self.scale_output = AdaptiveOutputScale(dim=dim, cond_dim=cond_dim, rngs=rngs)

    def __call__(self, x, cond, mask):
        """
        x: input representation of shape [b, n, dim]
        cond: condition of shape [b, n, cond_dim]
        mask: binary mask, shape [b, n]
        """
        x = self.adaln(x, cond, mask)
        x = self.transition(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]

class MHBiasedAttnBlock(nnx.Module):
    """Multi-head pair-biased attention block with adaptive layernorm/scaling and transition layer."""
    def __init__(
        self, 
        node_dim: int, 
        cond_dim: int, 
        pair_dim: int, 
        n_heads: int, 
        residual_mha: bool,
        residual_transition: bool,
        parallel_mha_transition: bool,
        use_qkln: bool,
        rngs: nnx.Rngs, 
        expension_factor: int = 4,
    ):
        self.parallel = parallel_mha_transition
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadBiasAttn_ADALN(
            node_dim=node_dim, 
            cond_dim=cond_dim,
            pair_dim=pair_dim,
            n_heads=n_heads,
            use_qkln=use_qkln, 
            rngs=rngs,
        )
        self.transition = TransitionADALN(dim=node_dim, cond_dim=cond_dim, rngs=rngs, expension_factor=expension_factor)
    
    def _apply_mha(self, x, pair_repr, cond, mask):
        x_attn = self.mhba(x, pair_repr, cond, mask)
        if self.residual_mha:
            x_attn = x + x_attn
        return x_attn * mask[..., None]
    
    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x + x_tr
        return x_tr * mask[..., None]

    def __call__(self, node_feats, pair_repr, cond, mask):
        """
        node_feats: [b, n, node_dim]
        pair_repr: [b, n, n, pair_dim]
        cond: [b, n, cond_dim]
        mask: [b, n]
        """
        x = node_feats * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_repr, cond, mask) + self._apply_transition(x, cond, mask)
        else:
            x = self._apply_mha(x, pair_repr, cond, mask)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]