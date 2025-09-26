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
import numpy as np
import flax, optax
from flax import nnx
import os, glob, compress_pickle

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def dump_ckpt(
    params, ema_params, opt_state, train_rng, step, epoch, wandb_id,
    fname,
    ):
    compress_pickle.dump({
            'params': params,
            'ema_params': ema_params,
            'opt_state': opt_state,
            'train_rng': train_rng,
            'step': step,
            'epoch': epoch,
            'wandb_id': wandb_id
        },
        open(fname, 'wb'),
        compression='gzip'
    )
    return


def load_ckpt_from_pkl(fname, model, use_ema=True):
    # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
    # Adopted from https://flax.readthedocs.io/en/latest/guides/checkpointing.html#restore-checkpoints
    ckpt = compress_pickle.load(open(fname, 'rb'), compression='gzip')
    if use_ema:
        params = ckpt['ema_params']
    else:
        params = ckpt['params']
    graphdef, abstract_state = nnx.split(nnx.eval_shape(lambda: model))
    model = nnx.merge(graphdef, params)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print('Loaded checkpoint from %d, number of parameters: %d'%(ckpt['epoch'], total_params))
    print('NNX State restored, number of parameters: ', total_params)
    return model

def load_most_recent_ckpt_for_restart(ckpt_dir):
    # Load the latest checkpoint from the directory
    # Adopted from https://flax.readthedocs.io/en/latest/guides/checkpointing.html#restore-checkpoints
    ckpt_files = glob.glob(ckpt_dir + '/*.pkl')
    if len(ckpt_files) == 0:
        print('No checkpoint found in %s, train from scratch...'%ckpt_dir)
        ckpt = {
            'params': None, 'ema_params': None, 'opt_state': None, 
            'train_rng': None, 'step': 0, 'epoch': 0,
            'wandb_id': None
        }
        return ckpt
    ckpt_files.sort(key=os.path.getctime)
    latest_ckpt = ckpt_files[-1]
    ckpt = compress_pickle.load(open(latest_ckpt, 'rb'), compression='gzip')
    params, epoch = ckpt['params'], ckpt['epoch']
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print('Loaded checkpoint from %d, number of parameters: %d'%(epoch, total_params))
    print('NNX State restored, number of parameters: ', total_params)
    return ckpt


def load_most_recent_ckpt_for_restart_cluster(ckpt_dir):
    # Load the latest checkpoint from the directory
    # Adopted from https://flax.readthedocs.io/en/latest/guides/checkpointing.html#restore-checkpoints
    ckpt_files = glob.glob(ckpt_dir + '/restart.pkl')
    if len(ckpt_files) == 0:
        print('No checkpoint found in %s, train from scratch...'%ckpt_dir)
        ckpt = {
            'params': None, 'ema_params': None, 'opt_state': None, 
            'train_rng': None, 'step': 0, 'epoch': 0,
            'wandb_id': None
        }
        return ckpt
    try:
        ckpt = compress_pickle.load(open(ckpt_files[0], 'rb'), compression='gzip')
    except:
        print('Restart ckpt saved with error, try load from storage...')
        ckpt_files = glob.glob(ckpt_dir.replace('restart', 'storage') + '/*.pkl')
        if len(ckpt_files) == 0:
            print('No checkpoint found in %s, train from scratch...'%ckpt_dir)
            ckpt = {
                'params': None, 'ema_params': None, 'opt_state': None, 
                'train_rng': None, 'step': 0, 'epoch': 0,
                'wandb_id': None
            }
            return ckpt
        ckpt_files.sort(key=os.path.getctime)
        latest_ckpt = ckpt_files[-1]
        ckpt = compress_pickle.load(open(latest_ckpt, 'rb'), compression='gzip')
    params, epoch = ckpt['params'], ckpt['epoch']
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    print('Loaded checkpoint from %d, number of parameters: %d'%(epoch, total_params))
    print('NNX State restored, number of parameters: ', total_params)
    return ckpt


def set_lr_scheduler_configs(config):
    """
    Unpack the learning rate scheduler config and set the learning rate scheduler.
    """
    lr = config['learning_rate'] if 'learning_rate' in config else 0.0002
    init_value = config['init_value'] if 'init_value' in config else 1e-6
    warmup_steps = config['warmup_steps'] if 'warmup_steps' in config else 10_000
    decay_steps = config['decay_steps'] if 'decay_steps' in config else 1_000_000
    end_value = config['end_value'] if 'end_value' in config else 1e-6
    lr_scheduler_configs = {
        'learning_rate': lr,
        'init_value': init_value,
        'warmup_steps': warmup_steps,
        'decay_steps': decay_steps,
        'end_value': end_value
    }
    return lr_scheduler_configs