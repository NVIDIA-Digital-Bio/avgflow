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

import os, glob, time, shutil
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.training.optimizer import _opt_state_variables_to_state
from nn.model import ConformerFlowTransformer
from dataloader.jnp_dataloader import ReflowLoader
from flow_matcher.flow_matcher import FlowMatcher
from utils.model_utils import tree_stack, load_most_recent_ckpt_for_restart_cluster, dump_ckpt, load_ckpt_from_pkl
import optax
import wandb, compress_pickle, signal
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec as P
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import yaml, argparse

def main(config):
    # read config file
    print(config)
    # Load data and instantiate the dataloader
    data_root_dir = config['data_root_dir']
    data_file_name = config['data_file_name']
    data_path = os.path.join(data_root_dir, data_file_name)
    data = pickle.load(open(data_path, 'rb'))
    n_val_samples = config['validation']['n_val_samples']
    train_data, val_data = data[:len(data)-n_val_samples], data[len(data)-n_val_samples:]

    # Create dataloader
    dataloader = ReflowLoader(
        [d['features'] for d in train_data], 
        [d['x0s'] for d in train_data], [d['x1s'] for d in train_data], 
        max_n=config['model']['max_n'], batchsize=config['model']['batchsize'],
        shuffle=True, drop_last=False
    )
    print('Loaded %d graphs for training'%len(train_data[0]))
    val_loader = ReflowLoader(
        [d['features'] for d in val_data], 
        [d['x0s'] for d in val_data], [d['x1s'] for d in val_data],
        max_n=config['model']['max_n'], batchsize=config['model']['batchsize'],
        shuffle=False, drop_last=False
    )
    print('Loaded %d graphs for validation'%len(val_data[0]))

    # Define the model
    rngs = nnx.Rngs(0)
    max_ne = config['model']['max_n']**2

    model = ConformerFlowTransformer(
        max_n=config['model']['max_n'], 
        node_raw_feat_dim=config['model']['feat_dim'], 
        pe_dim=config['model']['pe_dim'], 
        graph_feat_dim=config['model']['graph_feat_dim'], 
        cond_dim=config['model']['cond_dim'], 
        n_edge_type=config['model']['n_edge_type'], 
        edge_type_emb_dim=config['model']['edge_type_emb_dim'], 
        node_dim=config['model']['node_dim'], 
        n_heads=config['model']['n_heads'], 
        n_layers=config['model']['n_layers'],
        rngs=rngs,
        n_registers=config['model']['n_registers']
    )

    # Load the pretrained model for reflow finetuning
    pretrained_ckpt_path = config['pretrained_ckpt']
    print('Loading pretrained model from %s for reflow finetuning...'%(pretrained_ckpt_path), flush=True)
    model = load_ckpt_from_pkl(pretrained_ckpt_path, model, use_ema=config['load_ema'])

    params = nnx.state(model, nnx.Param)
    total_params = sum([np.prod(x.shape) for x in jax.tree.leaves(params)], 0)
    size_in_million = f"{round(total_params/1e6)}M"
    print('Total parameters:', total_params, flush=True)

    # Define optimizer
    device_count = jax.local_device_count()
    learning_rate_configs = set_lr_scheduler_configs(config['flow_matcher']['learning_rate_scheduler'])
    assert config['flow_matcher']['target_mode'] == 'linear', 'Only linear target is supported for reflow finetuning'
    flow_matcher = FlowMatcher(
        model=model,
        learning_rate_configs=learning_rate_configs,
        target_mode=config['flow_matcher']['target_mode'], # 'linear'
        loss_mode=config['flow_matcher']['loss_mode'], # 'x1'
        self_cond=config['flow_matcher']['self_cond'], # False
        scale_lr_by_num_devices=config['flow_matcher']['scale_lr_by_num_devices'], # False
        time_schedule=config['flow_matcher']['time_schedule'], # 'uniform' or 'mixed'
    )

    if config['flow_matcher']['time_schedule'] == 'distill':
        postfix = 'distill'
    else:
        postfix = 'reflow'
    run_name = f"{config['pretrained_ckpt'].split('/')[0]}_{postfix}"

    # Create the checkpoint directories
    os.makedirs(config['checkpoint_dir']+f'/{run_name}', exist_ok=True)
    os.makedirs(config['checkpoint_dir']+f'/{run_name}/restart', exist_ok=True)
    os.makedirs(config['checkpoint_dir']+f'/{run_name}/storage', exist_ok=True)
    # Copy config file to checkpoint directory
    config_dest = os.path.join(config['checkpoint_dir'], run_name, os.path.basename(config['config_path']))
    shutil.copy2(config['config_path'], config_dest)
    print(f"Copied config file to {config_dest}", flush=True)

    # Load the model from the checkpoint or train from scratch
    checkpoint = load_most_recent_ckpt_for_restart_cluster(config['checkpoint_dir']+f'/{run_name}/restart')
    model, params, ema_params, train_rng, train_steps, epoch = flow_matcher.initialize_model_and_optimizer(model, checkpoint)
    starting_epoch = epoch+1
    print('Starting from epoch %d, step %d'%(starting_epoch, train_steps+1), flush=True)
    beta = config['training']['ema_beta']
    
    available_devices = jax.devices()
    print(f"Device count: {device_count}, Available devices: {available_devices}", flush=True)
    mesh = Mesh(mesh_utils.create_device_mesh((device_count,)), axis_names=("ax"))
    sharding = NamedSharding(mesh, P("ax"))

    save_ep = []
    assert len(config['training']['save_every_n_epoch']) - 1 == len(config['training']['save_interval_change_threshold'])
    save_interval_change_threshold = [0]+config['training']['save_interval_change_threshold']+[config['training']['n_epochs']+1]
    for i in range(1, len(save_interval_change_threshold)):
        save_ep += list(range(save_interval_change_threshold[i-1], save_interval_change_threshold[i], config['training']['save_every_n_epoch'][i-1]))
    print(f"Will save checkpoints at epoch: {save_ep}", flush=True)

    use_wandb = config['wandb']['use_wandb']
    if use_wandb:
        assert config['wandb']['entity'] is not None, "WandB entity is not set"
        if checkpoint['wandb_id'] is not None:
            wandb_id = checkpoint['wandb_id']
        else:
            now = datetime.now()
            wandb_id = now.strftime("%Y_%m_%d_%H_%M_%S")
        logger=wandb.init(
            entity=config['wandb']['entity'], project=config['wandb']['project'], 
            name=run_name, id=wandb_id, resume="allow"
        )
    else:
        wandb_id, logger = None, None

    val_every_n_epoch = config['validation']['val_every_n_epoch']
    for ep in range(starting_epoch, config['training']['n_epochs']+1):
        MSE = 0
        nbatches = 0
        batches = []
        for i, batch in enumerate(dataloader):
            batches.append(batch)
            if len(batches) == device_count:
                shared_batch = jax.device_put(tree_stack(batches), sharding)
                params, train_rng = flow_matcher.train_step(
                    model=model,
                    batches=shared_batch, 
                    key=train_rng
                )
                ema_params= jax.tree.map(lambda x, y: beta*x + (1-beta)*y, params, ema_params)
                MSE_ = flow_matcher.metrics.compute()
                flow_matcher.metrics.reset()
                MSE += MSE_
                nbatches += 1
                train_steps += 1
                batches = []
                if train_steps % config['training']['log_every_n_step'] == 0:
                    print(f"--------Step {train_steps}, MSE: {MSE_}--------", flush=True)
                    if use_wandb:
                        logger.log(data={"step_MSE": MSE_, "train_step": train_steps})
                
        MSE = MSE / nbatches

        if val_every_n_epoch > 0 and ep % val_every_n_epoch == 0:
            val_MSE = 0
            val_nbatches = 0
            val_batches = []
            for i, batch in enumerate(val_loader):
                val_batches.append(batch)
                if len(val_batches) == device_count:
                    shared_batch = jax.device_put(tree_stack(val_batches), sharding)
                    flow_matcher.val_step(model, shared_batch, train_rng)
                    val_MSE_ = flow_matcher.val_metrics.compute()
                    flow_matcher.val_metrics.reset()
                    val_MSE += val_MSE_
                    val_nbatches += 1
                    val_batches = []
            val_MSE = val_MSE / val_nbatches
            if use_wandb:
                logger.log(data={"val_MSE": val_MSE, "epoch": ep})

        if use_wandb:
            logger.log(data={"MSE": MSE, "epoch": ep})
        print(f"==============Epoch {ep}, MSE: {MSE}==============", flush=True)
        ## Save ckpt for restart
        dump_ckpt(
            params, ema_params, _opt_state_variables_to_state(flow_matcher.optimizer.opt_state), 
            train_rng, flow_matcher.optimizer.step, ep, wandb_id,
            config['checkpoint_dir']+f'/{run_name}/restart/restart.pkl'
        )

        if ep in save_ep:
            dump_ckpt(
                params, ema_params, _opt_state_variables_to_state(flow_matcher.optimizer.opt_state), 
                train_rng, flow_matcher.optimizer.step, ep, wandb_id,
                config['checkpoint_dir']+f'/{run_name}/storage/ep_%d_ckpt.pkl'%ep
            )
            print('Additional checkpoint saved at epoch %d'%ep, flush=True)
        
    print('Finished training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    config['config_path'] = args.config
    main(config)
    

