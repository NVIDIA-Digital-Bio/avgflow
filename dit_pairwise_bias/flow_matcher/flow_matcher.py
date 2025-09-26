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
import flax, optax, diffrax
from flax import nnx
from flax.nnx.training.optimizer import _update_opt_state, _opt_state_variables_to_state
from functools import partial
import einops
import time
from .minimal_avg_flow import avg_flow

class FlowMatcher:
    def __init__(
        self, 
        model: nnx.Module, 
        learning_rate_configs: dict,
        target_mode: str,
        loss_mode: str,
        self_cond: bool=False,
        scale_lr_by_num_devices: bool=True,
        time_schedule: str='uniform', # uniform, or mixed (uniform beta 0.02)
    ):
        self.num_devices = jax.local_device_count()
        self.lr_scheduler_configs = learning_rate_configs
        self.lr = learning_rate_configs['learning_rate']
        self.lr = self.lr*self.num_devices if scale_lr_by_num_devices else self.lr
        self.lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate_configs['init_value'],
            peak_value=self.lr,
            warmup_steps=learning_rate_configs['warmup_steps'],
            decay_steps=learning_rate_configs['decay_steps'],
            end_value=learning_rate_configs['end_value'],
        )
        self.optimizer = nnx.Optimizer(
            model, 
            optax.adamw(learning_rate=self.lr_schedule) # use AdamW
        )
        self.metrics = nnx.metrics.Average('MSE')
        self.val_metrics = nnx.metrics.Average('val_MSE')
        self.max_n = model.max_n
        self.target_mode = target_mode
        self.loss_mode = loss_mode
        self.self_cond = self_cond
        self.time_schedule = time_schedule
        
    def initialize_model_and_optimizer(self, model: nnx.Module, ckpt: dict):
        """
        Load the most recent checkpoint for the model and optimizer.
        Args:
            model (nnx.Module): The model to be trained.
            ckpt_dir (dict): The directory where the checkpoint is saved.
        Returns:
            model (nnx.Module): The updated model.
            params (flax.core.frozen_dict.FrozenDict): The model parameters.
            ema_params (flax.core.frozen_dict.FrozenDict): The EMA model parameters.
            train_rng (jax.Array): The random number generator key.
            step (int): The training step.
            epoch (int): The training epoch.
        """
        if ckpt['params'] is None:
            params = nnx.state(model, nnx.Param)
            ema_params = nnx.state(model, nnx.Param)
            train_rng = jax.random.key(2025)
            step = -1
            epoch = -1
            return model, params, ema_params, train_rng, step, epoch
        params, ema_params, opt_state = ckpt['params'], ckpt['ema_params'], ckpt['opt_state'], 
        train_rng, step, epoch = ckpt['train_rng'], ckpt['step'], ckpt['epoch']
        # Load the model parameters and optimizer state
        graphdef, abstract_state = nnx.split(nnx.eval_shape(lambda: model))
        model = nnx.merge(graphdef, params)

        self.optimizer = nnx.Optimizer(
            model, 
            optax.adamw(learning_rate=self.lr_schedule) # use AdamW
        )
        _update_opt_state(
            self.optimizer.opt_state, 
            opt_state
        )
        self.optimizer.step = step
        step = int(step.value)

        return model, params, ema_params, train_rng, step, epoch

    def predict_flow(self, model: nnx.Module, batch: dict):
        preds = model(
            xt=batch['xt'],
            xt_hat=batch['xt_hat'],
            atomic_features=batch['node_feats'],
            laplacian_pos_enc=batch['pe'],
            graph_feats=batch['graph_feats'],
            senders=batch['senders'],
            receivers=batch['receivers'],
            edge_types=batch['edge_types'],
            time=batch['t'],
            mask=batch['mask']
        )

        return preds * batch['mask'][..., None]
    
    def v_to_x1_pred(self, pred_flow: jnp.ndarray, xt: jnp.ndarray, t: jnp.ndarray, mask: jnp.ndarray):
        """
        Convert flow prediction to x1 prediction given xt, time, and mask.
        Args:
            pred_flow (jnp.ndarray): The predicted flow array with shape (batch_size, max_n, 3).
            xt (jnp.ndarray): The current state array with shape (batch_size, max_n, 3).
            t (jnp.ndarray): The time array with shape (batch_size,).
            mask (jnp.ndarray): The mask array with shape (batch_size, max_n).
        Returns:
            x1_pred (jnp.ndarray): The predicted next state array with shape (batch_size, max_n, 3).
        """
        t_expand = jnp.repeat(jnp.repeat(t[..., None], self.max_n, axis=1)[..., None], 3, axis=-1)
        x1_pred = xt + (jnp.ones_like(t_expand, dtype=jnp.float32) - t_expand) * pred_flow
        return x1_pred * mask[..., None]

    def interpolate(self, batch: dict, rng: jax.Array, model: nnx.Module):
        """
        Get the time (t), interpolation (xt), ground truth flow (ut) depending on the mode.
        Args:
            batch (dict): The input batch dictionary.
            rng (jax.Array): The random number generator.
            model (nnx.Module): The model used to generate self-conditioning input.
        Returns:
            batch (dict): The updated batch dictionary with the t, xt, ut, and xt_hat.
        """
        mode = self.target_mode
        assert mode in ['linear', 'avgflow'], f"Invalid target mode: {mode}"
        rng, sc_rng, t_rng = jax.random.split(rng, num=3)
        batchsize = batch['x1'].shape[0] 
        x1, mask = batch['x1'], batch['mask']

        # Sample time from uniform (or mixed) and x0 from normal
        t = jax.random.uniform(rng, (batchsize,)) * 0.999
        if self.time_schedule == 'mixed':
            t_beta = jax.random.beta(rng, 1.9, 1.0, (batchsize,)) * 0.999
            t = jnp.where(jax.random.uniform(t_rng, (batchsize,))<0.1, t, t_beta)
        elif self.time_schedule == 'distill':
            t = jnp.zeros((batchsize,))

        # To accomodate for the case where x0 is provided (reflow or any alignment)
        if "x0" in batch:
            x0 = batch['x0'] * mask[..., None]
        else:
            x0 = jax.random.normal(rng, (batchsize, self.max_n, 3)) * mask[..., None]
        t_expand = jnp.repeat(jnp.repeat(t[..., None], self.max_n, axis=1)[..., None], 3, axis=-1)

        if mode == 'linear':
            # linear interpolant
            xt = x0 + t_expand * (x1 - x0)
            xt = xt * mask[..., None]
            ut = x1 - x0
        elif mode == 'avgflow':
            # use the minimal avgflow solver
            # solve the xt using avgflow (integration interpolant)
            xt = _avgflow_xt_solver(t, x0, x1) * mask[..., None]
            ut = jax.vmap(avg_flow)(t, xt, x1[:, None, :, :])
        else:
            raise NotImplementedError(f"Invalid target mode: {mode}")
        
        batch['t'] = t
        batch['xt'] = xt
        batch['ut'] = ut
        # This is for self-conditioning, not used for now so zeroed out
        batch['xt_hat'] = jnp.zeros_like(xt, dtype=jnp.float32)
        if self.self_cond:
            # When self-conditioning, we condition the model on it's own prediction with 50% probability
            sc_decider = (jax.random.uniform(sc_rng, ()) * jnp.ones_like(xt)) > 0.5
            pred_flow = self.predict_flow(model, batch)
            if self.loss_mode == 'x1':
                sc_input = self.v_to_x1_pred(pred_flow, batch['xt'], batch['t'], batch['mask'])
            else:
                sc_input = pred_flow
            batch['xt_hat'] = jnp.where(sc_decider, sc_input, batch['xt_hat'])
        return batch
    
    def loss_fn(self, model: nnx.Module, batches: dict):
        """
        Compute the loss function given the batch.
        Args:
            model (nnx.Module): The model to be trained.
            batches (dict): The input batch dictionary. 
                For multi-GPU training, each value should be of shape (n_devices, batch_size, max_n, ...).
        Returns:
            loss (jnp.ndarray): The computed loss.
        """
        mode = self.loss_mode
        assert mode in ['x1', 'ut'], f"Invalid mode: {mode}, either predict x1 or ut"
        if mode == 'x1' and self.target_mode == 'avgflow':
            raise ValueError("We should not use avgflow and x1 prediction at the same time.")

        def _inner_vmap(batch):
            ut_pred = self.predict_flow(model, batch)
            if mode == 'x1':
                pred = self.v_to_x1_pred(ut_pred, batch['xt'], batch['t'], batch['mask'])
                target = batch['x1']
            elif mode == 'ut':
                pred = ut_pred
                target = batch['ut']
            squared_loss = jnp.sum((pred - target) ** 2, -1) * batch['mask'] # (batch_size, max_n)
            n_atoms = jnp.sum(batch['mask'], axis=-1) # (batch_size,)
            # Regular MSE loss scaled by the number of atoms
            loss = jnp.mean(jnp.sum(squared_loss, axis=-1) / n_atoms)
            return loss
        
        loss = jax.vmap(
            _inner_vmap, in_axes=(0)
        )(batches)
        loss= loss.mean()
        return loss

    def train_step(
        self, 
        model: nnx.Module,
        batches: dict, 
        key: jax.Array
    ):
        """
        Train for a single step.
        Args:
            model (nnx.Module): The model to be trained.
            batches (dict): The input batch dictionary.
            key (jax.Array): The random number generator key.
        Returns:
            params (flax.core.frozen_dict.FrozenDict): The updated model parameters.
            key (jax.Array): The updated random number generator
        """
        return _train_step(
            self.interpolate, 
            self.loss_fn, 
            model, 
            self.optimizer, 
            self.metrics, 
            batches, 
            key
        )
    
    def val_step(
        self, 
        model: nnx.Module,
        batches: dict, 
        key: jax.Array
    ):
        """
        Validation for a single step.
        Args:
            model (nnx.Module): The model to be trained.
            batches (dict): The input batch dictionary.
            key (jax.Array): The random number generator key.
        Returns:
            metrics (dict): The updated metrics dictionary.
            key (jax.Array): The updated random number generator
        """
        return _val_step(
            self.interpolate, 
            self.loss_fn, 
            model, 
            self.val_metrics, 
            batches, 
            key
        )

    def generate(
        self, model, batch, key,
        num_steps=100, 
        t_schedule='uniform', t_p=2.0,
        solver='euler'
    ):  
        if self.self_cond:
            raise NotImplementedError("Self-conditioning is not supported yet during generation.")

        return _generate_ode(
            model=model, batch=batch, key=key, 
            num_steps=num_steps, t_schedule=t_schedule, t_p=t_p, 
            ODEsolver=solver
        )


@partial(nnx.jit, static_argnums=(0,1))
def _train_step(interpolate_fn, loss_fn, model, optimizer, metrics, batches, key):
    print('Compiling _train_step...')
    key, sampling_key = jax.random.split(key, 2)
    
    # Sample for the batches
    batches = jax.vmap(
        interpolate_fn, 
        in_axes=(0, None, None)
    )(batches, sampling_key, model)

    # Compute loss and gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(model, batches)

    # Update optimizer and metrics
    metrics.update(MSE=loss)
    optimizer.update(grads)

    # Get updated parameters
    params = nnx.state(model, nnx.Param)
    return params, key

@partial(nnx.jit, static_argnums=(0,1))
def _val_step(interpolate_fn, loss_fn, model, metrics, batches, key):
    print('Compiling _val_step...')
    # Sample for the batches
    batches = jax.vmap(
        interpolate_fn, 
        in_axes=(0, None, None)
    )(batches, key, model)

    # Compute loss
    loss = loss_fn(model, batches)

    # Update metrics
    metrics.update(val_MSE=loss)
    return
    
def get_log_schedule(p, nsteps):
    cutoff = 0.999
    t = 1.0 - jnp.flip(jnp.logspace(-p, 0, nsteps)) 
    t = t - jnp.min(t)
    t = t / jnp.max(t)
    t = t*cutoff
    t = t.at[-1].set(1.0)
    return t

def gt(t, threshold=0.99):
    scale = 1.0 / (t + 0.01)
    return jnp.where(t < threshold, scale, 0.0)

# Solve the ODE for the xt when using avgflow mode
def _avgflow_xt_solver(t, x0, x1):
    """
    Solve the ODE for the xt when using avgflow mode.
    Args:
        t (jnp.ndarray): The time array. [b, ]
        x0 (jnp.ndarray): Noise coordinates. [b, n, 3]
        x1 (jnp.ndarray): Clean samples. [b, n, 3]
    Returns:
        xt (jnp.ndarray): The xt by solving ground truth avgflow ODE. [b, n, 3]
    """
    print('Compiling _avgflow_xt_solver...')
    num_steps = 20 # fixed number of steps for now
    batchsize = t.shape[0]
    
    # Create ts_steps for all batch elements
    ts_steps = jax.vmap(lambda t_i: jnp.linspace(0, t_i, num_steps))(t)  # [batchsize, num_steps]
    # Calculate time deltas
    dts = ts_steps[:, 1:] - ts_steps[:, :-1]  # [batchsize, num_steps-1]
    # Reshape ts_steps and dts to [num_steps-1, batchsize]
    ts_steps, dts = jnp.transpose(ts_steps[:, :-1], (1, 0)), jnp.transpose(dts, (1, 0))

    # Euler step
    def scan_body(carry, step_data):
        xt = carry
        t, dt = step_data
        u = jax.vmap(avg_flow)(t, xt, x1[:, None, :, :])
        next_xt = xt + u*dt[:, None, None]
        return next_xt, None
    
    # Prepare scan input
    scan_input = (ts_steps, dts)  # Use all steps except last for t, and all deltas
    
    # Run the scan
    final_xt, _ = jax.lax.scan(
        scan_body,
        init=x0,
        xs=scan_input
    )
    
    return final_xt
    


@partial(nnx.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def _generate_ode(
    model, batch, key, 
    num_steps, t_schedule, t_p=2.0,
    ODEsolver='euler', rtol=1e-5, atol=1e-4, 
):
    print('Compiling _generate...')
    # Currently only support uniform and log t-schedule
    if t_schedule == 'uniform':
        ts=jnp.linspace(0, 1, num_steps+1)
    elif t_schedule == 'log':
        assert num_steps >= 2, "num_steps must be >= 2 for log t-schedule"
        ts=get_log_schedule(p=t_p, nsteps=num_steps+1)
    else:
        raise NotImplementedError
    dt0 = None
    
    if ODEsolver == 'euler':
        solver = diffrax.Euler() 
        # stepsize_controller = diffrax.ConstantStepSize()
        stepsize_controller = diffrax.StepTo(ts=ts)
        saveat = diffrax.SaveAt(ts=ts[:])
    elif ODEsolver == 'tsit5':
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=ts[-1:])
    else:
        raise NotImplementedError

    mask = batch['mask'] # [b, n]
    b, n = mask.shape
    x0 = jax.random.normal(key, (b, n, 3))
    x0 = x0 * mask[..., None]

    def flow(t, x, args):
        data = args
        b, n = data['mask'].shape
        x = x * data['mask'][..., None]
        u = model(
            xt=x,
            xt_hat = jnp.zeros_like(x), # Todo: maybe add self-conditioning here
            atomic_features=data['node_feats'],
            laplacian_pos_enc=data['pe'],
            graph_feats=data['graph_feats'],
            senders=data['senders'],
            receivers=data['receivers'],
            edge_types=data['edge_types'],
            time=t*jnp.ones((b,)),
            mask=data['mask']
        )
        return u
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(flow), solver, 0.0, 1.0, dt0, x0, 
        batch, saveat=saveat,
        stepsize_controller=stepsize_controller,
    )
    traj = sol.ys
    return traj