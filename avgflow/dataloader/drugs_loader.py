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
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm

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

def load_single_geom_drugs_reflow(data):
    root, smiles = data
    featpath = root+'mols/%s.pkl'%smiles
    confpath = root+'reflow_confs/%s'%featpath.split('/')[-1].replace('.pkl', '.npz')
    with open(featpath, "rb") as f:
        features = compress_pickle.load(f, compression="gzip")
    conf = np.load(confpath, allow_pickle=True)
    x0s = conf['x0']
    x1s = conf['x1']
    return features, x0s, x1s, smiles

def kabsch_np(x, y):
    n = x.shape[0]
    assert x.shape == (n, 3)
    assert y.shape == (n, 3)
    x = x - np.mean(x, axis=0)
    y = y - np.mean(y, axis=0)
    h = x.T @ y
    u, _, vh = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vh.T @ u.T))
    R = vh.T @ np.diag(np.array([1, 1, d])) @ u.T
    return R

def kabsch_align(data_pair):
    x, y = data_pair
    R = kabsch_np(x, y)
    return x @ R.T

class SingleLoader:
    def __init__(
        self,
        graph: jraph.GraphsTuple,
        conformers: np.ndarray,
        max_n: int,
        batchsize: int = 8,
        shuffle: bool = True,
    ):  
        # assert len(graphs) == len(conformers) == len(weights)
        graph = graph._replace(
            receivers=graph.receivers.astype('int32'), 
            senders=graph.senders.astype('int32'),
        )
        self.graph = graph
        self.conformers = conformers
        self.shuffle = shuffle
        max_ne = max_n**2
        self.max_n, self.max_ne = max_n, max_ne
        self.batchsize = batchsize

        self.num_nodes = graph.nodes['features'].shape[0]

        self.node_feats = self._pad(graph.nodes['features'], max_n)
        self.node_feats = jnp.repeat(self.node_feats[None, ...], batchsize, axis=0)

        self.pe = self._pad(graph.nodes['laplacian_pe'], max_n)
        self.pe = jnp.repeat(self.pe[None, ...], batchsize, axis=0)

        self.mask = self._pad(jnp.ones((self.num_nodes, )), max_n)
        print('mask shape before repeat:', self.mask.shape)
        self.mask = jnp.repeat(self.mask[None, ...], batchsize, axis=0)
        print('mask shape after repeat:', self.mask.shape)

        print(self._pad(graph.senders, max_ne).shape, graph.senders.shape)

        self.senders = jnp.repeat(self._pad(graph.senders, max_ne)[None, ...], batchsize, axis=0)
        self.receivers = jnp.repeat(self._pad(graph.receivers, max_ne)[None, ...], batchsize, axis=0)

        self.edge_types = self._pad(jnp.argmax(graph.edges['features'], axis=1), max_ne)
        self.edge_types = jnp.repeat(self.edge_types[None, ...], batchsize, axis=0)
        print(f"node_feats shape: {self.node_feats.shape}")
        print(f"pe shape: {self.pe.shape}")
        print(f"mask shape: {self.mask.shape}")
        print(f"senders shape: {self.senders.shape}")
        print(f"receivers shape: {self.receivers.shape}")
        print(f"edge_types shape: {self.edge_types.shape}")

    def get_batch(self, dkey=None):
        if dkey is not None:
            # indices = np.random.choice(self.conformers.shape[0], size=self.batchsize, replace=False)
            # indices = np.arange(self.batchsize)
            indices = jax.random.choice(dkey, self.conformers.shape[0], (self.batchsize,), replace=False)
            # indices = jax.random.randint(dkey, (self.batchsize,), 0, self.conformers.shape[0])
            sampled_conformers = self.conformers[indices]
            # x0 = np.random.normal(0, 1, (self.batchsize, self.max_n, 3))
            x0 = jax.random.normal(dkey, (self.batchsize, self.max_n, 3)) 
            x0 = jnp.array(x0, dtype=jnp.float32) * self.mask[..., None]
        else:
            indices = np.arange(self.batchsize)
            sampled_conformers = self.conformers[indices]
            x0 = jax.random.normal(jax.random.PRNGKey(42), (self.batchsize, self.max_n, 3))
            x0 = jnp.array(x0, dtype=jnp.float32) * self.mask[..., None]
        # Center the conformers
        sampled_conformers = jax.vmap(lambda x: x - jnp.mean(x, axis=0))(sampled_conformers)
        sampled_conformers = self._pad(sampled_conformers, self.max_n)
        t = jnp.array(np.random.uniform(0, 1, (self.batchsize,)) * 0.999, dtype=jnp.float32)
        t_expand = jnp.repeat(jnp.repeat(t[..., None], self.max_n, axis=1)[..., None], 3, axis=-1)
        xt = x0 + t_expand * (sampled_conformers-x0)
        v = sampled_conformers - x0
        out_dict = {
            'xt': xt * self.mask[..., None],
            't': t,
            'node_feats': self.node_feats,
            'pe': self.pe,
            'mask': self.mask,
            'senders': self.senders,
            'receivers': self.receivers,
            'edge_types': self.edge_types,
            't': t, 
            'flow': v * self.mask[..., None]
        }
        return out_dict

    def get_batch_official(self):
        indices = np.random.choice(self.conformers.shape[0], size=self.batchsize, replace=True)
        sampled_conformers = self.conformers[indices]
        # Center the conformers
        sampled_conformers = jax.vmap(lambda x: x - jnp.mean(x, axis=0))(sampled_conformers)
        sampled_conformers = self._pad(sampled_conformers, self.max_n)

        out_dict = {
            'x1': sampled_conformers * self.mask[..., None],
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


class GeomDrugsLoader:
    def __init__(
        self,
        graphs: jraph.GraphsTuple,
        conformers: np.ndarray,
        # smiles: list,
        max_n: int,
        batchsize: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        assert len(graphs) == len(conformers)
        graphs = [
            g._replace(
                receivers=g.receivers.astype('int32'), 
                senders=g.senders.astype('int32'),
            ) for g in graphs
        ]
        self.graphs, self.conformers = [], []
        for i in range(len(graphs)):
            if graphs[i].nodes['features'].shape[0] > max_n:
                continue
            # if graphs[i].nodes['features'].shape[1] != 75:
            #     continue
            self.graphs.append(graphs[i])
            self.conformers.append(conformers[i])
        # self.graphs = graphs
        # self.conformers = conformers
        # self.smiles = smiles

        print('Actual number of graphs:', len(self.graphs))
        self.shuffle = shuffle
        self.drop_last = drop_last
        max_ne = max_n**2
        self.max_n, self.max_ne = max_n, max_ne
        self.batchsize = batchsize
        self.num_devices = jax.local_device_count()
        self.total_batchsize = self.batchsize * self.num_devices
    
    def __iter__(self):
        sampled_graphs = []
        for i in range(len(self.graphs)):
            conf_idx = np.random.choice(len(self.conformers[i]))
            sampled_conf = self.conformers[i][conf_idx]
            sampled_graphs.append(self.add_pos_to_graph(self.graphs[i], sampled_conf))

        if self.shuffle:
            random.shuffle(sampled_graphs)

        if self.drop_last:
            num_batches = len(sampled_graphs) // self.total_batchsize   
            sampled_graphs = sampled_graphs[:num_batches*self.batchsize]
        else:
            padding = self.total_batchsize - len(sampled_graphs) % self.total_batchsize
            sampled_graphs += sampled_graphs[-padding:]
        
        for batch in batch_iterator(sampled_graphs, self.batchsize):
            yield self._process_batched_graphs(batch)
    

    def _pad(self, x, N):
        if len(x.shape) == 1:
            return jnp.pad(x, (0, N-x.shape[0])) if x.shape[0] < N else x
        if len(x.shape) == 2:
            return jnp.pad(x, ((0, N-x.shape[0]), (0, 0))) if x.shape[0] < N else x
        if len(x.shape) == 3:
            return jnp.pad(x, ((0, 0), (0, N-x.shape[1]), (0, 0))) if x.shape[1] < N else x
        raise ValueError("Only support 1D, 2D, or 3D array")
    
    def _process_batched_graphs(self, batch):
        """
        batch: list of jraph.GraphsTuple
        """
        batched_features = [self._pad_single_graph_with_conformer(g) for g in batch] # List of dict
        combined_dict = {key: jnp.stack([d[key] for d in batched_features], axis=0) for key in batched_features[0].keys()}
        return combined_dict
        
    
    def _pad_single_graph_with_conformer(self, graph: jraph.GraphsTuple):
        """
        For a single graph in a batch, pad the graph to max_n nodes and max_ne edges
        Args:
            graph: jraph.GraphsTuple
        """
        max_n = self.max_n
        max_ne = self.max_ne
        n_node = graph.nodes['features'].shape[0]
        # Pad node related features
        node_feats = self._pad(graph.nodes['features'], self.max_n) # [1, max_n, feat_dim]
        pe = self._pad(graph.nodes['laplacian_pe'], self.max_n) # [1, max_n, pe_dim]
        x1 = self._pad(graph.nodes['x1'], self.max_n) # [1, max_n, 3]
        mask = self._pad(jnp.ones((n_node, )), self.max_n) # [1, max_n]
        # Pad edge related features
        senders = self._pad(graph.senders, self.max_ne) # [1, max_ne]
        receivers = self._pad(graph.receivers, self.max_ne) # [1, max_ne]
        edge_types = self._pad(jnp.argmax(graph.edges['features'], axis=1), self.max_ne) # [1, max_ne]

        out_dict = {
            'x1': x1,
            'node_feats': node_feats,
            'pe': pe,
            'mask': mask,
            'senders': senders,
            'receivers': receivers,
            'edge_types': edge_types,
            'graph_feats': jnp.zeros((5, ), dtype=jnp.float32), # Not using this for now
        }
        return out_dict

    def add_pos_to_graph(self, graph: jraph.GraphsTuple, pos: np.ndarray) -> jraph.GraphsTuple:
        graph = graph._replace(nodes=graph.nodes | {"x1": pos-np.mean(pos, axis=0)})
        return graph


class KabschLoader(GeomDrugsLoader):
    def __init__(
        self,
        graphs: jraph.GraphsTuple,
        conformers: np.ndarray,
        max_n: int,
        batchsize: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        super(KabschLoader, self).__init__(graphs, conformers, max_n, batchsize, shuffle, drop_last)
    
    def __iter__(self):
        sampled_graphs = []
        sampled_conformer_list = []
        sampled_x0_list = []
        for i in range(len(self.graphs)):
            conf_idx = np.random.choice(len(self.conformers[i]))
            sampled_conf = self.conformers[i][conf_idx]
            x0 = np.random.normal(0, 1, sampled_conf.shape)
            sampled_conformer_list.append(sampled_conf)
            sampled_x0_list.append(x0)
        
        print("Aligning conformers...")
        with Pool(processes=32) as p:
            aligned_x0 = list(
                p.imap(
                    kabsch_align, 
                    zip(sampled_x0_list, sampled_conformer_list),
                    )
            )
        for i in range(len(self.graphs)):
            sampled_graphs.append(
                self.add_pos_to_graph(
                    self.graphs[i], aligned_x0[i], sampled_conformer_list[i]
                )
            )

        if self.shuffle:
            random.shuffle(sampled_graphs)

        if self.drop_last:
            num_batches = len(sampled_graphs) // self.total_batchsize   
            sampled_graphs = sampled_graphs[:num_batches*self.batchsize]
        else:
            padding = self.total_batchsize - len(sampled_graphs) % self.total_batchsize
            sampled_graphs += sampled_graphs[-padding:]
        
        for batch in batch_iterator(sampled_graphs, self.batchsize):
            yield self._process_batched_graphs(batch)
        
    
    def _pad_single_graph_with_conformer(self, graph: jraph.GraphsTuple):
        """
        For a single graph in a batch, pad the graph to max_n nodes and max_ne edges
        Args:
            graph: jraph.GraphsTuple
        """
        max_n = self.max_n
        max_ne = self.max_ne
        n_node = graph.nodes['features'].shape[0]
        # Pad node related features
        node_feats = self._pad(graph.nodes['features'], self.max_n) # [1, max_n, feat_dim]
        pe = self._pad(graph.nodes['laplacian_pe'], self.max_n) # [1, max_n, pe_dim]
        x1 = self._pad(graph.nodes['x1'], self.max_n) # [1, max_n, 3]
        x0 = self._pad(graph.nodes['x0'], self.max_n) # [1, max_n, 3]
        mask = self._pad(jnp.ones((n_node, )), self.max_n) # [1, max_n]
        # Pad edge related features
        senders = self._pad(graph.senders, self.max_ne) # [1, max_ne]
        receivers = self._pad(graph.receivers, self.max_ne) # [1, max_ne]
        edge_types = self._pad(jnp.argmax(graph.edges['features'], axis=1), self.max_ne) # [1, max_ne]

        out_dict = {
            'x1': x1,
            'x0': x0,
            'node_feats': node_feats,
            'pe': pe,
            'mask': mask,
            'senders': senders,
            'receivers': receivers,
            'edge_types': edge_types,
            'graph_feats': jnp.zeros((5, ), dtype=jnp.float32), # Not using this for now
        }
        return out_dict
    
    def add_pos_to_graph(self, graph: jraph.GraphsTuple, x0: np.ndarray, x1: np.ndarray) -> jraph.GraphsTuple:
        graph = graph._replace(nodes=graph.nodes | {"x0": x0, "x1": x1-np.mean(x1, axis=0)})
        return graph

class ReflowLoader(GeomDrugsLoader):
    def __init__(
        self,
        graphs: jraph.GraphsTuple,
        x0s: np.ndarray,
        x1s: np.ndarray,
        max_n: int,
        batchsize: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        assert len(graphs) == len(x0s) == len(x1s)
        graphs = [
            g._replace(
                receivers=g.receivers.astype('int32'),
                senders=g.senders.astype('int32'),
            ) for g in graphs
        ]
        self.graphs, self.x0s, self.x1s = [], [], []
        for i in range(len(graphs)):
            if graphs[i].nodes['features'].shape[0] > max_n:
                continue
            self.graphs.append(graphs[i])
            self.x0s.append(x0s[i])
            self.x1s.append(x1s[i])

        print('Actual number of graphs:', len(self.graphs))
        self.shuffle = shuffle
        self.drop_last = drop_last
        max_ne = max_n**2
        self.max_n, self.max_ne = max_n, max_ne
        self.batchsize = batchsize
        self.num_devices = jax.local_device_count()
        self.total_batchsize = self.batchsize * self.num_devices

    def __iter__(self):
        sampled_graphs = []
        sampled_x1_list = []
        sampled_x0_list = []
        for i in range(len(self.graphs)):
            conf_idx = np.random.choice(len(self.x1s[i]))
            sampled_x1 = self.x1s[i][conf_idx]
            sampled_x0 = self.x0s[i][conf_idx]
            sampled_graphs.append(
                self.add_pos_to_graph(self.graphs[i], sampled_x0, sampled_x1)
            )

        if self.shuffle:
            random.shuffle(sampled_graphs)

        if self.drop_last:
            num_batches = len(sampled_graphs) // self.total_batchsize   
            sampled_graphs = sampled_graphs[:num_batches*self.batchsize]
        else:
            padding = self.total_batchsize - len(sampled_graphs) % self.total_batchsize
            sampled_graphs += sampled_graphs[-padding:]
        
        for batch in batch_iterator(sampled_graphs, self.batchsize):
            yield self._process_batched_graphs(batch)
        
    
    def _pad_single_graph_with_conformer(self, graph: jraph.GraphsTuple):
        """
        For a single graph in a batch, pad the graph to max_n nodes and max_ne edges
        Args:
            graph: jraph.GraphsTuple
        """
        max_n = self.max_n
        max_ne = self.max_ne
        n_node = graph.nodes['features'].shape[0]
        # Pad node related features
        node_feats = self._pad(graph.nodes['features'], self.max_n) # [1, max_n, feat_dim]
        pe = self._pad(graph.nodes['laplacian_pe'], self.max_n) # [1, max_n, pe_dim]
        x1 = self._pad(graph.nodes['x1'], self.max_n) # [1, max_n, 3]
        x0 = self._pad(graph.nodes['x0'], self.max_n) # [1, max_n, 3]
        mask = self._pad(jnp.ones((n_node, )), self.max_n) # [1, max_n]
        # Pad edge related features
        senders = self._pad(graph.senders, self.max_ne) # [1, max_ne]
        receivers = self._pad(graph.receivers, self.max_ne) # [1, max_ne]
        edge_types = self._pad(jnp.argmax(graph.edges['features'], axis=1), self.max_ne) # [1, max_ne]

        out_dict = {
            'x1': x1,
            'x0': x0,
            'node_feats': node_feats,
            'pe': pe,
            'mask': mask,
            'senders': senders,
            'receivers': receivers,
            'edge_types': edge_types,
            'graph_feats': jnp.zeros((5, ), dtype=jnp.float32), # Not using this for now
        }
        return out_dict
    
    def add_pos_to_graph(self, graph: jraph.GraphsTuple, x0: np.ndarray, x1: np.ndarray) -> jraph.GraphsTuple:
        graph = graph._replace(nodes=graph.nodes | {"x0": x0, "x1": x1-np.mean(x1, axis=0)})
        return graph


if __name__ == "__main__":
    data_root_dir = '/home/zhonglinc/storage/research/conformer/data/geom/drugs_noH/'
    smiles_splits = np.load(data_root_dir+'split/split0_clean.npy', allow_pickle=True)
    train_smiles, _, _ = smiles_splits
    train_smiles = train_smiles[4]
    print(train_smiles)

    graph, conformers, _ = load_single_geom_drugs((data_root_dir, train_smiles))
    print(graph.edges)
    # for i in range(len(graph.senders)):
    #     print(graph.senders[i], '->', graph.receivers[i])
    print(graph.edges['features'].shape)
    print(len(graph.senders), graph.nodes['features'].shape)
    print(graph.nodes.keys())
    print(jnp.argmax(graph.edges['features'], axis=1).shape)
    # print(graph.senders)
    dataloader = SingleLoader(graph, conformers, max_n=64)
    @jax.jit
    def get_batch_data():
        return dataloader.get_batch()
    batch = get_batch_data()
    print(batch['t'])