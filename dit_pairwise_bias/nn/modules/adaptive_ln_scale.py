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

from typing import Any, Tuple
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange

class Identity(nnx.Module):
    def __call__(self, x):
        return x


class AdaptiveLayerNorm(nnx.Module):
    def __init__(self, dim: int, cond_dim: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(num_features=dim, rngs=rngs, use_scale=False, use_bias=False)
        self.norm_cond = nnx.LayerNorm(num_features=cond_dim, rngs=rngs)
        self.to_gamma = nnx.Sequential(
            nnx.Linear(cond_dim, dim, use_bias=False, rngs=rngs),
            nnx.sigmoid
        )
        self.to_beta = nnx.Linear(cond_dim, dim, use_bias=False, rngs=rngs)

    def __call__(self, x, cond, mask):
        """
        x: input representation of shape [*, dim]
        cond: condition of shape [*, cond_dim]
        mask: binary mask, shape [*]
        """
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)
        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        output = normed * gamma + beta
        return output * mask[..., None]

class AdaptiveOutputScale(nnx.Module):
    def __init__(self, dim: int, cond_dim: int, rngs: nnx.Rngs, adaln_zero_bias_init_value: float=-2.0):
        adaln_zero_gamma_linear = nnx.Linear(
            cond_dim, dim, 
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.constant(value=adaln_zero_bias_init_value)
        )
        self.to_adaln_zero_gamma = nnx.Sequential(
            adaln_zero_gamma_linear,
            nnx.sigmoid
        )

    def __call__(self, x, cond, mask):
        """
        x: input representation of shape [*, dim]
        cond: condition of shape [*, cond_dim]
        mask: binary mask, shape [*]
        """
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma * mask[..., None]