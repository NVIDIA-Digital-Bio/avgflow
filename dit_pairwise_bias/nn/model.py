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
import warnings
from typing import Any, Tuple
from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange
from .modules.feature_factory import *
from .modules.pair_bias_attn import *
from .modules.full_attention_block import *

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

class ConformerFlowTransformer(nnx.Module):
    def __init__(
        self, 
        max_n: int, 
        node_raw_feat_dim: int, 
        pe_dim: int, 
        graph_feat_dim: int, 
        cond_dim: int, 
        n_edge_type: int, 
        edge_type_emb_dim: int, 
        node_dim: int, 
        n_heads: int, 
        n_layers: int,
        rngs: nnx.Rngs,
        n_registers: int = 0,
    ):
        assert n_edge_type == 5, "Only 5 edge types are supported for now"
        if not is_power_of_two(edge_type_emb_dim+2):
            warnings.warn("Suggest to use power of 2 for edge_type_emb_dim+2")
        self.n_layers = n_layers
        self.cond_dim = cond_dim
        self.max_n = max_n

        key = rngs.params()
        assert n_registers >= 0, "Number of registers should be non-negative"
        self.n_registers = n_registers
        if self.n_registers > 0:
            self.registers = nnx.Param(
                jax.random.normal(
                    key, 
                    shape=(self.n_registers, node_dim), 
                    dtype=jnp.float32
                ) / 20
            ) # [n_registers, node_dim]
        else:
            self.registers = None


        self.atomic_repr_block = AtomicRepresentation(max_n, 3 + 3 + node_raw_feat_dim + pe_dim, node_dim, rngs, use_layer_norm=False)
        self.condition_block = ConditionBlock(max_n, graph_feat_dim, cond_dim, rngs, use_layer_norm=False)
        self.cond_pair_block = PairwiseBlock(n_edge_type, edge_type_emb_dim, cond_dim, rngs)

        self.cond_transition_c1 = Transition(dim=cond_dim, rngs=rngs, expension_factor=2, layer_norm=False)
        self.cond_transition_c2 = Transition(dim=cond_dim, rngs=rngs, expension_factor=2, layer_norm=False)

        self.main_trunk = []
        for _ in range(n_layers):
            self.main_trunk.append(
                MHBiasedAttnBlock(
                    node_dim=node_dim,
                    cond_dim=cond_dim,
                    pair_dim=edge_type_emb_dim+2,
                    n_heads=n_heads, 
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_qkln=True, 
                    rngs=rngs, 
                    expension_factor=4
                )
            )
        self.coords_3d_decoder = nnx.Sequential(
            nnx.LayerNorm(num_features=node_dim, rngs=rngs),
            nnx.Linear(in_features=node_dim, out_features=3, rngs=rngs, use_bias=False)
        )
        
    def _extend_with_registers(self, node_repr, pair, cond, mask):
        """
        Extends the sequence representation, pair representation, mask and indices with registers.

        Args:
            - seqs: sequence representation, shape [b, n, dim_token]
            - pair: pair representation, shape [b, n, n, dim_pair]
            - mask: binary mask, shape [b, n]
            - cond_seq: tensor of shape [b, n, dim_cond]

        Returns:
            All elements above extended with registers / zeros.
        """
        if self.n_registers == 0:
            return node_repr, pair, cond, mask
        r = self.n_registers
        b, n, _ = node_repr.shape
        
        # Extend node representation with self.registers
        reg_expanded = self.registers[None, ...] # [1, r, node_dim]
        reg_expanded = jnp.repeat(reg_expanded, b, axis=0) # [b, r, node_dim]
        node_repr = jnp.concatenate([reg_expanded, node_repr], axis=1) # [b, r+n, node_dim]

        # Extend mask
        true_tensor = jnp.ones((b, r), dtype=mask.dtype)
        mask = jnp.concat([true_tensor, mask], axis=1) # [b, r+n]

        # Extend pair representation with zeros; pair has shape [b, n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        # [b, n, n, pair_dim] -> [b, r+n, n, pair_dim]
        zero_pad_top = jnp.zeros((b, r, n, pair.shape[-1]), dtype=pair.dtype)
        pair = jnp.concat([zero_pad_top, pair], axis=1) # [b, r+n, n, pair_dim]
        # [b, r+n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        zero_pad_left = jnp.zeros((b, r+n, r, pair.shape[-1]), dtype=pair.dtype)
        pair = jnp.concat([zero_pad_left, pair], axis=2)

        # Extend cond
        cond_pad = jnp.zeros((b, r, cond.shape[-1]), dtype=cond.dtype)
        cond = jnp.concatenate([cond_pad, cond], axis=1)

        return node_repr, pair, cond, mask

    def _undo_registers(self, node_repr, pair, mask):
        """
        Removes the registers from the sequence representation, pair representation and mask.

        Args:
            - seqs: sequence representation, shape [b, r+n, node_dim]
            - pair: pair representation, shape [b, r+n, r+n, pair_dim]
            - mask: binary mask, shape [b, r+n]

        Returns:
            All elements above with the registers removed.
        """
        if self.n_registers == 0:
            return node_repr, pair, mask
        r = self.n_registers
        b, n, _ = node_repr.shape
        node_repr = node_repr[:, r:, :]
        pair = pair[:, r:, r:, :]
        mask = mask[:, r:]
        return node_repr, pair, mask


    def __call__(self, xt, xt_hat, atomic_features, laplacian_pos_enc, graph_feats, senders, receivers, edge_types, time, mask):
        node_repr = self.atomic_repr_block(xt, xt_hat, atomic_features, laplacian_pos_enc, mask)
        cond = self.condition_block(time, graph_feats, mask)
        cond = self.cond_transition_c2(self.cond_transition_c1(cond, mask), mask)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        conditioned_pair_repr = self.cond_pair_block(xt, xt_hat, senders, receivers, edge_types, time, mask)

        # Apply registers
        node_repr, conditioned_pair_repr, cond, mask = self._extend_with_registers(node_repr, conditioned_pair_repr, cond, mask)

        for i in range(self.n_layers):
            node_repr = self.main_trunk[i](node_repr, conditioned_pair_repr, cond, mask)

        # Undo registers
        node_repr, conditioned_pair_repr, mask = self._undo_registers(node_repr, conditioned_pair_repr, mask)
        flow = self.coords_3d_decoder(node_repr) * mask[..., None]
        return flow