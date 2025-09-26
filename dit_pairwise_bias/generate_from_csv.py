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

import os, glob, time
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx
from flax.nnx.training.optimizer import _opt_state_variables_to_state
from nn.model import ConformerFlowTransformer
from dataloader.jnp_dataloader import SingleGenerationLoader
from flow_matcher.flow_matcher import FlowMatcher
from utils.model_utils import tree_stack, load_ckpt_from_pkl, set_lr_scheduler_configs
from utils.rdkit_utils import add_conformer_to_mol
import optax, diffrax
import wandb, pickle, compress_pickle, signal
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PositionalSharding
from jax.sharding import PartitionSpec as P
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
import yaml, argparse
import sys
sys.path.append('/home/zhonglinc/storage/research/avgflow_release/dit_pairwise_bias/data_preprocessing')
from preprocess import mol2features
from utils.rdkit_utils import post_hoc_flip_mol
from rdkit import Chem

def main(args, config):
    smiles_df = pd.read_csv(args.smiles_csv)
    smiles_list = smiles_df['SMILES'].tolist()
    n_confs_list = smiles_df['num_confs'].tolist()
    total_num_confs_in_csv = sum(n_confs_list)
    print('Generating %d conformers for %d molecules'%(total_num_confs_in_csv, len(smiles_list)))
    # read config file
    print('Generation config:')
    print(config)


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

    ckpt_path = config['checkpoint_path']
    use_ema = config['use_ema']
    model = load_ckpt_from_pkl(
        fname=ckpt_path,
        model=model,
        use_ema=use_ema
    )


    # Initialize the flow matcher
    device_count = jax.local_device_count()
    learning_rate_configs = set_lr_scheduler_configs(config['flow_matcher']['learning_rate_scheduler'])
    flow_matcher = FlowMatcher(
        model=model,
        learning_rate_configs=learning_rate_configs,
        target_mode=config['flow_matcher']['target_mode'], # 'linear'
        loss_mode=config['flow_matcher']['loss_mode'], # 'x1'
        self_cond=config['flow_matcher']['self_cond'], # False
    )

    run_name = config['run_name']
    n_steps = config['n_steps']
    solver = config['solver']
    # Defining generation batchsize
    batchsize = config['model']['batchsize']
    t_schedule = config['t_schedule']
    t_p = config['t_p']
    featurize_mode = config['featurize_mode']
    total_num_flip = 0
    key = jax.random.key(config['random_seed'])

    pbar = tqdm(total=total_num_confs_in_csv)
    # Start generation
    for i, row in smiles_df.iterrows():
        # print(f'Generating conformers for {row["SMILES"]} ({i+1}/{len(smiles_df)})')
        smiles = row['SMILES']
        n_confs = row['num_confs']
        base_mol = Chem.MolFromSmiles(smiles)
        base_mol = Chem.AddHs(base_mol)
        graph = mol2features(base_mol, featurize_mode)
        # Get a copy of the molecule to generate jraph graphtuple and canonical smiles
        dataloader = SingleGenerationLoader(
            graph, 
            max_n=config['model']['max_n'], 
            batchsize=batchsize
        )
        n_atoms = graph['node_feats'].shape[0]
        num_batches, remaining = divmod(n_confs, batchsize)
        if remaining > 0 or num_batches == 0:
            num_batches += 1

        generated_mols = []
        # Generate conformers
        key, sample_key = jax.random.split(key)
        num_generated = 0
        # with tqdm(total=n_confs) as pbar:
        for _ in range(num_batches):
            batch = dataloader.get_batch()
            x1_pred = flow_matcher.generate(
                model=model, batch=batch, 
                key=sample_key, 
                num_steps=n_steps, 
                t_schedule=t_schedule, t_p=t_p,
                solver=solver
            ) # (frames, batchsize, n_atoms, 3) frames = 1 if only_save_last else n_steps
            for i in range(batchsize):
                traj = x1_pred[:, i, :n_atoms, :]
                coordinates = traj[-1]
                conf_mol = add_conformer_to_mol(base_mol, coordinates)
                # Post-hoc flip the generated molecule if SMILES does not match ground truth SMILES
                # Make sure to use the canonical SMILES without hydrogen for the ground truth SMILES
                conf_mol = post_hoc_flip_mol(gt_smis=smiles, gen_mol=conf_mol)
                generated_mols.append(conf_mol)
                num_generated += 1
                pbar.update(1)
                if num_generated == n_confs:
                    break
            key, sample_key = jax.random.split(key)

        # Create the destination directory
        dst_dir = config['destination_dir']
        dst_dir = os.path.join(config['destination_dir'], f'{run_name}_{solver}_{t_schedule}_{n_steps}step')
        os.makedirs(dst_dir, exist_ok=True)
        # Record the generation config
        gen_config = {
            'run_name': run_name,
            'solver': solver,
            't_schedule': t_schedule,
            'n_steps': n_steps,
            'checkpoint': ckpt_path,
            'ema': use_ema,
        }
        # The output is a dictionary with the following keys:
        # - smiles: the SMILES string of the molecule
        # - gen_config: the generation configuration
        # - mol_with_confs: a list of generated molecules
        output_dict = {
            'smiles': smiles,
            'gen_config': gen_config,
            'mol_with_confs': generated_mols,
        }
        # Save the output dictionary
        # Replace '/' with '_' in the SMILES string to avoid file name errors
        safe_smiles = smiles.replace('/', '_')
        dst = os.path.join(dst_dir, f'{safe_smiles}.pkl')
        pickle.dump(
            output_dict, 
            open(dst, 'wb')
        )
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    parser.add_argument("-s", "--smiles_csv", type=str, help='Path to the CSV file containing SMILES strings and number of conformers to be generated')
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    main(args, config)
    

