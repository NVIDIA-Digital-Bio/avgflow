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

class DataLoader:
    def __init__(
        self,
        graphs: list[dict], 
        conformers: list[np.ndarray],
        max_n: int,
        batchsize: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Base DataLoader class for training and validation

        Args:
            graphs: list of dicts, each dict contains the features of a molecule graph
            conformers: list of numpy arrays, each array contains the coordinates of all conformers of a molecule
            max_n: maximum number of nodes in a graph, will pad graphs to max_n nodes
            batchsize: batch size
            shuffle: whether to shuffle the data
            drop_last: whether to drop the last batch if it is not full
        """
        # Check if the number of graphs and conformers are the same
        assert len(graphs) == len(conformers)
        self.graphs, self.conformers = [], []

        # Filter out graphs that have more than max_n nodes
        n_dropped = 0
        for i in range(len(graphs)):
            if graphs[i]['node_feats'].shape[0] > max_n:
                n_dropped += 1
                continue
            self.graphs.append(graphs[i])
            self.conformers.append(conformers[i])
        print('Actual number of graphs:', len(self.graphs), '| Dropped for exceeding max_n:', n_dropped)
        self.shuffle = shuffle
        self.drop_last = drop_last
        max_ne = max_n**2
        self.max_n, self.max_ne = max_n, max_ne
        self.batchsize = batchsize
        self.num_devices = jax.local_device_count()
        self.global_batchsize = self.batchsize * self.num_devices
    
    def __iter__(self):
        sampled_graphs = []
        # Randomly sample a conformer for each molecule based on uniform distribution
        # Can certainly be do boltzmann distribution, just add boltzmann weights to random choice
        for i in range(len(self.graphs)):
            conf_idx = np.random.choice(len(self.conformers[i]))
            sampled_conf = self.conformers[i][conf_idx]
            sampled_graphs.append(self.add_pos_to_graph(self.graphs[i], sampled_conf))

        # Shuffle
        if self.shuffle:
            random.shuffle(sampled_graphs)

        if self.drop_last:
            # Drop last batch
            num_batches = len(sampled_graphs) // self.global_batchsize   
            sampled_graphs = sampled_graphs[:num_batches*self.batchsize]
        else:
            # Pad last batch
            padding = self.global_batchsize - len(sampled_graphs) % self.global_batchsize
            sampled_graphs += sampled_graphs[-padding:]
        
        for batch in batch_iterator(sampled_graphs, self.batchsize):
            yield self._process_batched_graphs(batch)
    

    def _pad(self, x, N):
        """
        Pad the array x to the maximum number of nodes N
        Args:
            x: numpy array
            N: int
        Returns:
            padded_x: jnp array
        """
        if len(x.shape) == 1:
            return jnp.pad(x, (0, N-x.shape[0])) if x.shape[0] < N else x
        if len(x.shape) == 2:
            return jnp.pad(x, ((0, N-x.shape[0]), (0, 0))) if x.shape[0] < N else x
        if len(x.shape) == 3:
            return jnp.pad(x, ((0, 0), (0, N-x.shape[1]), (0, 0))) if x.shape[1] < N else x
        raise ValueError("Only support 1D, 2D, or 3D array")
    
    def _process_batched_graphs(self, batch):
        """
        Process a batch of graphs
        Args:
            batch: list of dicts, each dict contains the features of a molecule graph
        Returns:
            combined_dict: dict, keys are the same as the dicts in the batch, values are jnp arrays
        """
        batched_features = [self._pad_single_graph_with_conformer(g) for g in batch] # List of dicts
        combined_dict = {key: jnp.stack([d[key] for d in batched_features], axis=0) for key in batched_features[0].keys()}
        return combined_dict
        
    
    def _pad_single_graph_with_conformer(self, graph: dict):
        """
        For a single graph in a batch, pad the graph to max_n nodes and max_ne edges
        Args:
            graph: dict, keys are 'node_feats', 'pe', 'x1', 'mask', 'senders', 'receivers', 'edge_types'
        """
        n_node = graph['node_feats'].shape[0]
        # Pad node related features
        node_feats = self._pad(graph['node_feats'], self.max_n) # [1, max_n, feat_dim]
        pe = self._pad(graph['pe'], self.max_n) # [1, max_n, pe_dim]
        x1 = self._pad(graph['x1'], self.max_n) # [1, max_n, 3]
        mask = self._pad(jnp.ones((n_node, )), self.max_n) # [1, max_n]
        # Pad edge related features
        senders = self._pad(graph['senders'], self.max_ne) # [1, max_ne]
        receivers = self._pad(graph['receivers'], self.max_ne) # [1, max_ne]
        edge_types = self._pad(jnp.argmax(graph['edge_types'], axis=1), self.max_ne) # [1, max_ne]

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

    def add_pos_to_graph(self, graph: dict, pos: np.ndarray) -> dict:
        """
        Add the conformer (x1 in flow-matching notation) to the graph. Center the conformer to the origin. 
        Args:
            graph: dict,
            pos: np.ndarray, shape (n_node, 3)
        """
        graph['x1'] = pos-np.mean(pos, axis=0)
        return graph



class ReflowLoader(DataLoader):
    def __init__(
        self,
        graphs: list,
        x0s: np.ndarray,
        x1s: np.ndarray,
        max_n: int,
        batchsize: int = 8,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize the ReflowLoader.
        Args:
            graphs: list of dicts containing graph features
            x0s: array of initial conformer positions
            x1s: array of target conformer positions
            max_n: maximum number of nodes per graph
            batchsize: batch size per device
            shuffle: whether to shuffle data
            drop_last: whether to drop last incomplete batch
        """
        assert len(graphs) == len(x0s) == len(x1s)
        self.graphs, self.x0s, self.x1s = [], [], []
        n_dropped = 0
        for i in range(len(graphs)):
            if len(graphs[i]['node_feats']) > max_n:
                n_dropped += 1
                continue
            self.graphs.append(graphs[i])
            self.x0s.append(x0s[i])
            self.x1s.append(x1s[i])

        print('Actual number of graphs:', len(self.graphs), '| Dropped for exceeding max_n:', n_dropped)
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
            sampled_graphs = sampled_graphs[:num_batches*self.total_batchsize]
        else:
            padding = self.total_batchsize - len(sampled_graphs) % self.total_batchsize
            sampled_graphs += sampled_graphs[-padding:]
        
        for batch in batch_iterator(sampled_graphs, self.batchsize):
            yield self._process_batched_graphs(batch)
        
    def _pad_single_graph_with_conformer(self, graph: dict):
        """Pad a single graph to max size.
        Args:
            graph: dict containing graph features
        Returns:
            dict of padded features
        """
        n_node = graph['node_feats'].shape[0]
        # Pad node related features
        node_feats = self._pad(graph['node_feats'], self.max_n)
        pe = self._pad(graph['pe'], self.max_n)
        x1 = self._pad(graph['x1'], self.max_n)
        x0 = self._pad(graph['x0'], self.max_n)
        mask = self._pad(jnp.ones((n_node, )), self.max_n)
        # Pad edge related features
        senders = self._pad(graph['senders'], self.max_ne)
        receivers = self._pad(graph['receivers'], self.max_ne)
        edge_types = self._pad(jnp.argmax(graph['edge_types'], axis=1), self.max_ne)

        out_dict = {
            'x1': x1,
            'x0': x0, 
            'node_feats': node_feats,
            'pe': pe,
            'mask': mask,
            'senders': senders,
            'receivers': receivers,
            'edge_types': edge_types,
            'graph_feats': jnp.zeros((5, ), dtype=jnp.float32),
        }
        return out_dict
    
    def add_pos_to_graph(self, graph: dict, x0: np.ndarray, x1: np.ndarray) -> dict:
        """Add conformer positions to graph.
        Args:
            graph: dict of graph features
            x0: initial conformer positions
            x1: target conformer positions
        Returns:
            dict with added positions
        """
        graph['x0'] = x0
        graph['x1'] = x1 - np.mean(x1, axis=0)
        return graph


class SingleGenerationLoader:
    def __init__(
        self,
        graph: dict,
        max_n: int,
        batchsize: int = 32,
    ):  
        """
        Initialize the SingleGenerationLoader.
        The output of this loader is always a batch of same molecule graph data.
        Args:
            graph: dict of graph features
            max_n: maximum number of nodes per graph
            batchsize: batch size per device
        """
        max_ne = max_n**2
        self.max_n, self.max_ne = max_n, max_ne
        self.batchsize = batchsize

        self.num_nodes = graph['node_feats'].shape[0]

        self.node_feats = self._pad(graph['node_feats'], max_n)
        self.node_feats = jnp.repeat(self.node_feats[None, ...], batchsize, axis=0)

        self.pe = self._pad(graph['pe'], max_n)
        self.pe = jnp.repeat(self.pe[None, ...], batchsize, axis=0)

        self.mask = self._pad(jnp.ones((self.num_nodes, )), max_n)
        self.mask = jnp.repeat(self.mask[None, ...], batchsize, axis=0)

        self.senders = jnp.repeat(self._pad(graph['senders'].astype('int32'), max_ne)[None, ...], batchsize, axis=0)
        self.receivers = jnp.repeat(self._pad(graph['receivers'].astype('int32'), max_ne)[None, ...], batchsize, axis=0)

        self.edge_types = self._pad(jnp.argmax(graph['edge_types'], axis=1), max_ne)
        self.edge_types = jnp.repeat(self.edge_types[None, ...], batchsize, axis=0)

    def get_batch(self):
        """
        Get a batch of data
        Returns:
            dict of batched data
        """
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
        """
        Pad the array x to the maximum number of nodes N
        Args:
            x: numpy array
            N: int
        Returns:
            padded_x: jnp array
        """
        if len(x.shape) == 1:
            return jnp.pad(x, (0, N-x.shape[0])) if x.shape[0] < N else x
        if len(x.shape) == 2:
            return jnp.pad(x, ((0, N-x.shape[0]), (0, 0))) if x.shape[0] < N else x
        if len(x.shape) == 3:
            return jnp.pad(x, ((0, 0), (0, N-x.shape[1]), (0, 0))) if x.shape[0] < N else x
        raise ValueError("Only support 1D, 2D, or 3D array")