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

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import os, glob
import random
import compress_pickle


def batch_iterator(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

def load_single_geom_drugs(data):
    root, smiles = data
    featpath = root+'mols/%s.pkl'%smiles
    confpath = root+'confs/%s'%featpath.split('/')[-1].replace('.pkl', '.npz')
    with open(featpath, "rb") as f:
        features = compress_pickle.load(f, compression="gzip")

    conf = np.load(confpath, allow_pickle=True)
    conformers = conf['pos']
    bweights = conf['scaled_boltzmann']
    return features, conformers, bweights, smiles

class SingleGenerationLoader:
    def __init__(
        self,
        graph: jraph.GraphsTuple,
        max_n: int,
        batchsize: int = 32,
    ):  
        graph = graph._replace(
            receivers=graph.receivers.astype('int32'), 
            senders=graph.senders.astype('int32'),
        )
        self.graph = graph
        max_ne = max_n**2
        self.max_n, self.max_ne = max_n, max_ne
        self.batchsize = batchsize

        self.num_nodes = graph.nodes['features'].shape[0]

        self.node_feats = self._pad(graph.nodes['features'], max_n)
        self.node_feats = jnp.repeat(self.node_feats[None, ...], batchsize, axis=0)

        self.pe = self._pad(graph.nodes['laplacian_pe'], max_n)
        self.pe = jnp.repeat(self.pe[None, ...], batchsize, axis=0)

        self.mask = self._pad(jnp.ones((self.num_nodes, )), max_n)
        self.mask = jnp.repeat(self.mask[None, ...], batchsize, axis=0)

        self.senders = jnp.repeat(self._pad(graph.senders, max_ne)[None, ...], batchsize, axis=0)
        self.receivers = jnp.repeat(self._pad(graph.receivers, max_ne)[None, ...], batchsize, axis=0)

        self.edge_types = self._pad(jnp.argmax(graph.edges['features'], axis=1), max_ne)
        self.edge_types = jnp.repeat(self.edge_types[None, ...], batchsize, axis=0)

    def get_batch(self):
        out_dict = {
            'node_feats': self.node_feats,
            'pe': self.pe,
            'mask': self.mask,
            'senders': self.senders,
            'receivers': self.receivers,
            'edge_types': self.edge_types,
            'graph_feats': jnp.zeros((self.batchsize, 5), dtype=jnp.float32),
        }
        return out_dict

    def _pad(self, x, N):
        if len(x.shape) == 1:
            return jnp.pad(x, (0, N-x.shape[0])) if x.shape[0] < N else x
        if len(x.shape) == 2:
            return jnp.pad(x, ((0, N-x.shape[0]), (0, 0))) if x.shape[0] < N else x
        if len(x.shape) == 3:
            return jnp.pad(x, ((0, 0), (0, N-x.shape[1]), (0, 0))) if x.shape[0] < N else x
        raise ValueError("Only support 1D, 2D, or 3D array")