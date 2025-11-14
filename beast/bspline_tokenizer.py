import os
import json
from functools import wraps
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import mp_pytorch.util as mp_utils
import numpy as np
import torch
from addict import Dict
from mp_pytorch.mp import MPFactory
from tqdm import tqdm

from beast.base_tokenizer import TokenizerBase
from beast.utils import continuous_to_discrete, denormalize_tensor, discrete_to_continuous, normalize_tensor


def autocast_float32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        device_type = None
        if args:
            maybe_device = getattr(args[0], "device", None)
            if isinstance(maybe_device, torch.device):
                device_type = maybe_device.type
            elif isinstance(maybe_device, str):
                try:
                    device_type = torch.device(maybe_device).type
                except (TypeError, RuntimeError):
                    device_type = None
        if device_type == "cuda" and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float32):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapped


class BSpline_Tokenizer(TokenizerBase):

    def __init__(self, num_dof=1, num_basis=10, duration=2 * torch.pi, seq_len=50, vocab_size=256,
                 degree_p=4, gripper_zero_order=True, gripper_indices=None,
                 init_cond_order=0, end_cond_order=0, init_pos = True,
                 use_bpe=False, device="cuda"):
        super().__init__()

        self.dt = 0.01  # 100 Hz, fixed for now
        
        # Automatically determine gripper_dof from gripper_indices
        if gripper_indices is None or not gripper_zero_order:
            gripper_indices = []
        self.gripper_indices = sorted(gripper_indices)
        
        if not gripper_zero_order or len(self.gripper_indices) == 0:
            self.gripper_dof = 0
        else:
            self.gripper_dof = len(self.gripper_indices)
            
        self.joint_dof = num_dof - self.gripper_dof
        
        # Create masks for joint and gripper indices
        all_indices = set(range(num_dof))
        gripper_indices_set = set(self.gripper_indices)
        self.joint_indices = sorted(list(all_indices - gripper_indices_set))
        self.bspline_config = Dict()
        self.bspline_config.mp_type = "uni_bspline"
        self.bspline_config.device = device
        self.bspline_config.num_dof = self.joint_dof
        self.bspline_config.tau = duration
        self.bspline_config.mp_args.num_basis = num_basis
        self.bspline_config.mp_args.degree_p = degree_p
        self.bspline_config.mp_args.init_condition_order = init_cond_order
        self.bspline_config.mp_args.end_condition_order = end_cond_order
        self.bspline_config.mp_args.dt = 0.01
        # self.bspline_config.mp_args.weights_scale = 0.01
        self.init_pos = init_pos

        self.mp = MPFactory.init_mp(**self.bspline_config)

        self.gripper_mp = None

        if gripper_zero_order and self.gripper_dof > 0:
            self.gripper_mp_config = Dict()
            self.gripper_mp_config.mp_type = "uni_bspline"
            self.gripper_mp_config.device = device
            self.gripper_mp_config.num_dof = self.gripper_dof
            self.gripper_mp_config.tau = duration
            self.gripper_mp_config.mp_args.num_basis = num_basis
            self.gripper_mp_config.mp_args.degree_p = 0
            self.gripper_mp = MPFactory.init_mp(**self.gripper_mp_config)
            print(
                f"Gripper MP initialized with {num_basis} basis functions for "
                f"{self.gripper_dof} DOFs at indices {self.gripper_indices}"
            )

        self.device = device
        self.num_dof = self.joint_dof + self.gripper_dof
        self.num_basis = num_basis

        self.vocab_size = vocab_size

        self.duration = duration
        self.seq_length = seq_len

        self.use_bpe = use_bpe

        self.times = mp_utils.tensor_linspace(0, duration, seq_len).to(device)

        self.register_buffer("w_min", -0.02 * torch.ones((num_dof * num_basis)))
        self.register_buffer("w_max", 0.02 * torch.ones((num_dof * num_basis)))
        self.vlm_vocab_size = None
        
        # Store config for serialization
        self._config = {
            'num_dof': num_dof,
            'num_basis': num_basis,
            'duration': float(duration),
            'seq_len': seq_len,
            'vocab_size': vocab_size,
            'degree_p': degree_p,
            'gripper_zero_order': gripper_zero_order,
            'gripper_indices': list(self.gripper_indices),
            'init_cond_order': init_cond_order,
            'end_cond_order': end_cond_order,
            'init_pos': init_pos,
            'use_bpe': use_bpe,
            'device': device,
        }
    
    
    # ===============================================
    #           - tokenizer preparation -          
    # ===============================================
    
    def update_vlm_vocab_size(self, vlm_vocab_size):
        self.vlm_vocab_size = vlm_vocab_size
    
    
    def fit_parameters(self, dataloader, max_samples=None, verbose=True):
        """
        Fit weight bounds based on dataloader samples.
        Now properly handles gripper indices at any position.
        """
        params = []

        sample_limit = max_samples if max_samples is not None else float("inf")

        iterator = tqdm(dataloader, desc="precomputing weight normalizer of MP", unit="batch") if verbose else dataloader

        sample_count = 0
        for batch in iterator:
            if "actions" not in batch:
                raise KeyError("Expected batch to contain an 'actions' entry.")

            act_chunks = batch["actions"][..., : self.num_dof].to(self.device)

            param = self.compute_weights(act_chunks).to("cpu").numpy()
            params.append(param)

            sample_count += param.shape[0]
            if sample_count >= sample_limit:
                if verbose:
                    print("Precomputed enough samples for weight normalizer of MP")
                break

        if not params:
            raise RuntimeError("No parameters were gathered from the dataloader.")

        params = np.concatenate(params, axis=0)

        params_min = np.quantile(params, 0.01, 0)
        params_max = np.quantile(params, 0.99, 0)

        params_min = torch.from_numpy(params_min).to(self.w_min.device)
        params_max = torch.from_numpy(params_max).to(self.w_max.device)

        self.w_min.copy_(params_min)
        self.w_max.copy_(params_max)
        
    
    # ===============================================
    #           - tokenizer serialization -          
    # ===============================================

    def get_config(self):
        """Return the configuration dictionary."""
        config = self._config.copy()
        if self.vlm_vocab_size is not None:
            config['vlm_vocab_size'] = self.vlm_vocab_size
        return config


    def state_dict(self):
        """
        Return state dict containing fitted parameters and configuration.
        This includes w_min, w_max after fit_parameters.
        """
        return {
            'config': self.get_config(),
            'w_min': self.w_min.cpu().numpy().tolist(),
            'w_max': self.w_max.cpu().numpy().tolist(),
            'vlm_vocab_size': self.vlm_vocab_size,
        }
    

    def load_state_dict(self, state_dict):
        """
        Load state dict containing fitted parameters.
        
        Args:
            state_dict: Dictionary with 'w_min', 'w_max', and optionally 'vlm_vocab_size'
        """
        if 'w_min' in state_dict:
            w_min = torch.tensor(state_dict['w_min'], dtype=torch.float32, device=self.device)
            self.w_min.copy_(w_min)

        if 'w_max' in state_dict:
            w_max = torch.tensor(state_dict['w_max'], dtype=torch.float32, device=self.device)
            self.w_max.copy_(w_max)
        
        if 'vlm_vocab_size' in state_dict and state_dict['vlm_vocab_size'] is not None:
            self.vlm_vocab_size = state_dict['vlm_vocab_size']
        
        print(f"✓ Loaded fitted parameters (w_min, w_max) with shape {self.w_min.shape}")


    def save_pretrained(self, save_directory):
        """
        Save tokenizer config and fitted parameters to directory.
        
        Args:
            save_directory: Path to directory where to save
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save state dict as JSON
        state = self.state_dict()
        config_path = save_directory / "beast_tokenizer_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✓ Saved tokenizer to {save_directory}")
        print(f"  - Config: {config_path}")


    @classmethod
    def from_pretrained(cls, pretrained_path, device=None):
        """
        Load tokenizer from pretrained directory.
        
        Args:
            pretrained_path: Path to directory containing tokenizer_config.json
            device: Device to load to (overrides saved config if provided)
        
        Returns:
            BSpline_Tokenizer instance with fitted parameters loaded
        """
        pretrained_path = Path(pretrained_path)
        config_path = pretrained_path / "beast_tokenizer_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load state dict
        with open(config_path, 'r') as f:
            state = json.load(f)
        
        config = state['config']
        
        # Override device if specified
        if device is not None:
            config['device'] = device
        
        # Create tokenizer instance
        print(f"✓ Loading tokenizer from {pretrained_path}")
        print(f"  - Config: num_dof={config['num_dof']}, num_basis={config['num_basis']}, "
              f"gripper_indices={config['gripper_indices']}")
        
        tokenizer = cls(**config)
        
        # Load fitted parameters
        tokenizer.load_state_dict(state)
        
        return tokenizer
    
    # ===============================================
    #              - tokenizer utils -          
    # ===============================================

    @torch.no_grad()
    @autocast_float32
    def compute_weights(self, demos):
        demos = demos.to(self.device)
        times = einops.repeat(self.times, 't -> b t', b=demos.shape[0])
        
        # Extract joint trajectories using joint_indices
        joint_demos = demos[..., self.joint_indices]
        weights = self.mp.learn_mp_params_from_trajs(times, joint_demos)['params']
        
        if self.gripper_mp is not None:
            # Extract gripper trajectories using gripper_indices
            gripper_demos = demos[..., self.gripper_indices]
            gripper_weights = self.gripper_mp.learn_mp_params_from_trajs(times, gripper_demos)['params']
            weights = torch.cat([weights, gripper_weights], dim=-1)
        
        return weights

    @torch.no_grad()
    def update_weights_bounds(self, demos):
        demos = demos.to(self.device)
        times = einops.repeat(self.times, 't -> b t', b=demos.shape[0])
        
        # Extract joint trajectories using joint_indices
        joint_demos = demos[..., self.joint_indices]
        weights = self.mp.learn_mp_params_from_trajs(times, joint_demos)['params']
        
        if self.gripper_mp is not None:
            # Extract gripper trajectories using gripper_indices
            gripper_demos = demos[..., self.gripper_indices]
            gripper_weights = self.gripper_mp.learn_mp_params_from_trajs(times, gripper_demos)['params']
            weights = torch.cat([weights, gripper_weights], dim=-1)
        
        self.w_min.copy_(weights.min(dim=0)[0])
        self.w_max.copy_(weights.max(dim=0)[0])
    @torch.no_grad()
    def update_weights_bounds_per_batch(self, weights):
        weights = weights.reshape(-1, self.num_dof * self.num_basis)
        batch_min = weights.min(dim=0)[0]
        batch_max = weights.max(dim=0)[0]
        smaller_mask = batch_min < (self.w_min - 1e-4)
        larger_mask = batch_max > (self.w_max + 1e-4)
        if torch.any(smaller_mask):
            self.w_min.masked_scatter_(smaller_mask, batch_min[smaller_mask])
        if torch.any(larger_mask):
            self.w_max.masked_scatter_(larger_mask, batch_max[larger_mask])

    def update_times(self, times):
        self.times = times.to(self.device)
        
    
    # ===============================================
    #           - tokenizer encoding -          
    # ===============================================

    @torch.no_grad()
    @autocast_float32
    def encode(self, trajs, update_bounds=False):

        trajs = trajs.to(self.device, dtype=torch.float32)
        times = einops.repeat(self.times, 't -> b t', b=trajs.shape[0])
        
        # Extract joint trajectories using joint_indices
        joint_trajs = trajs[..., self.joint_indices]
        params_dict = self.mp.learn_mp_params_from_trajs(times, joint_trajs)
        
        if self.gripper_mp is not None:
            # Extract gripper trajectories using gripper_indices
            gripper_trajs = trajs[..., self.gripper_indices]
            gripper_params_dict = self.gripper_mp.learn_mp_params_from_trajs(times, gripper_trajs)
            params_dict['params'] = torch.cat([params_dict['params'], gripper_params_dict['params']], dim=-1)
        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])
        
        unclampled_params = params_dict['params']
        params = torch.clamp(unclampled_params, min=self.w_min, max=self.w_max)
        tokens = continuous_to_discrete(params, min_val=self.w_min, max_val=self.w_max, num_bins=self.vocab_size)
        # tokens = einops.rearrange(tokens, 'b (d t) -> b t d', t=self.num_basis, d=self.num_dof)
        tokens = einops.rearrange(tokens, 'b (d t) -> b (t d)', t=self.num_basis, d=self.num_dof)
        return tokens, params_dict
    
    @torch.no_grad()
    @autocast_float32
    def encode_continuous(self, trajs, update_bounds=False):
        trajs = trajs.to(self.device, dtype=torch.float32)
        times = einops.repeat(self.times, 't -> b t', b=trajs.shape[0])
        
        # Extract joint trajectories using joint_indices
        joint_trajs = trajs[..., self.joint_indices]
        params_dict = self.mp.learn_mp_params_from_trajs(times, joint_trajs)
        
        if self.gripper_mp is not None:
            # Extract gripper trajectories using gripper_indices
            gripper_trajs = trajs[..., self.gripper_indices]
            gripper_params_dict = self.gripper_mp.learn_mp_params_from_trajs(times, gripper_trajs)
            params_dict['params'] = torch.cat([params_dict['params'], gripper_params_dict['params']], dim=-1)
        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])
        tokens = params_dict['params']
        tokens = normalize_tensor(tokens, w_min=self.w_min, w_max=self.w_max)
        tokens = einops.rearrange(tokens, 'b (d t) -> b (t d)', t=self.num_basis, d=self.num_dof)
        return tokens, params_dict

    # ===============================================
    #           - tokenizer LLM tokenization -          
    # ===============================================

    def tokens_to_llm_tokens(self, tokens):
        tokens = tokens.to(self.device)
        if len(tokens.shape) == 3:
            tokens = einops.rearrange(tokens, 'b t d -> b (t d)')
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab size is not set.")
        llm_tokens = self.vlm_vocab_size - 1 - tokens
        return llm_tokens

    def llm_tokens_to_mp_tokens(self, llm_tokens):
        if self.vlm_vocab_size is None:
            raise ValueError("VLM vocab is not set.")
        tokens = self.vlm_vocab_size - 1 - llm_tokens
        if len(tokens.shape) == 2:
            tokens = einops.rearrange(tokens, 'b (t d) -> b t d', t=self.num_basis, d=self.num_dof)
        return tokens
    
    # ===============================================
    #            - tokenizer decoding -          
    # ===============================================

    def reconstruct_from_llm_tokens(self, llm_tokens, times=None, **kwargs):
        tokens = self.llm_tokens_to_mp_tokens(llm_tokens)
        return self.reconstruct_traj(tokens, times=times, **kwargs)

    def decode(self, tokens):
        tokens = tokens.to(self.device)
        if tokens.dim() == 3:
            tokens = einops.rearrange(tokens, 'b t d -> b (t d)')
        elif tokens.dim() != 2:
            raise ValueError(f"Unexpected token shape {tokens.shape}")

        tokens = einops.rearrange(tokens, 'b (t d) -> b (d t)', t=self.num_basis, d=self.num_dof)
        params = discrete_to_continuous(tokens, min_val=self.w_min, max_val=self.w_max, num_bins=self.vocab_size)
        return params

    @torch.no_grad()
    @autocast_float32
    def reconstruct_traj(self, tokens, times=None, **kwargs):
        # params = self.decode(tokens.reshape(-1, self.num_dof * self.num_basis))
        params = self.decode(tokens)
        if times is None:
            times = einops.repeat(self.times, 't -> b t', b=params.shape[0])
        if self.init_pos and kwargs.get("init_p") is not None:
            _params = einops.rearrange(params, "b (d t) -> b t d", t=self.num_basis, d=self.num_dof)
            # Update only joint initial positions
            for i, joint_idx in enumerate(self.joint_indices):
                _params[:, 0, i] = kwargs["init_p"][:, joint_idx]
            params = einops.rearrange(_params, "b t d -> b (d t)")
        
        # Reconstruct joint trajectories
        joint_params = params[..., :self.joint_dof * self.num_basis]
        self.mp.update_inputs(times=times, params=joint_params)
        joint_pos = self.mp.get_traj_pos()
        
        # Initialize full position tensor
        batch_size = joint_pos.shape[0]
        time_steps = joint_pos.shape[1]
        pos = torch.zeros(batch_size, time_steps, self.num_dof, device=joint_pos.device, dtype=joint_pos.dtype)
        
        # Place joint positions at correct indices
        for i, joint_idx in enumerate(self.joint_indices):
            pos[..., joint_idx] = joint_pos[..., i]
        
        if self.gripper_mp is not None:
            # Reconstruct gripper trajectories
            gripper_params = params[..., self.joint_dof * self.num_basis:]
            self.gripper_mp.update_inputs(times=times, params=gripper_params)
            gripper_pos = self.gripper_mp.get_traj_pos()
            
            # Place gripper positions at correct indices
            for i, gripper_idx in enumerate(self.gripper_indices):
                pos[..., gripper_idx] = gripper_pos[..., i]
        
        return pos

    @torch.no_grad()
    def reconstruct_traj_continuous(self, params, times=None, **kwargs):
        params = params.to(self.device)
        if len(params.shape) == 3:
            params = einops.rearrange(params, 'b t d -> b (t d)')
        if params.shape[-1] != self.num_basis * self.num_dof:
            raise ValueError(
                f"Token dimension {params.shape[-1]} does not match expected {self.num_basis * self.num_dof}."
            )
        params = einops.rearrange(params, 'b (t d) -> b (d t)', t=self.num_basis, d=self.num_dof)
        params = denormalize_tensor(params, w_min=self.w_min, w_max=self.w_max)
        if times is None:
            times = einops.repeat(self.times, 't -> b t', b=params.shape[0])
        if self.init_pos and kwargs.get("init_p") is not None:
            _params = einops.rearrange(params, "b (d t) -> b t d", t=self.num_basis, d=self.num_dof)
            # Update only joint initial positions
            for i, joint_idx in enumerate(self.joint_indices):
                _params[:, 0, i] = kwargs["init_p"][:, joint_idx]
            params = einops.rearrange(_params, "b t d -> b (d t)")
        
        # Reconstruct joint trajectories
        joint_params = params[..., :self.joint_dof * self.num_basis]
        self.mp.update_inputs(times=times, params=joint_params)
        joint_pos = self.mp.get_traj_pos()
        
        # Initialize full position tensor
        batch_size = joint_pos.shape[0]
        time_steps = joint_pos.shape[1]
        pos = torch.zeros(batch_size, time_steps, self.num_dof, device=joint_pos.device, dtype=joint_pos.dtype)
        
        # Place joint positions at correct indices
        for i, joint_idx in enumerate(self.joint_indices):
            pos[..., joint_idx] = joint_pos[..., i]
        
        if self.gripper_mp is not None:
            # Reconstruct gripper trajectories
            gripper_params = params[..., self.joint_dof * self.num_basis:]
            self.gripper_mp.update_inputs(times=times, params=gripper_params)
            gripper_pos = self.gripper_mp.get_traj_pos()
            
            # Place gripper positions at correct indices
            for i, gripper_idx in enumerate(self.gripper_indices):
                pos[..., gripper_idx] = gripper_pos[..., i]
        
        return pos
    
    
    # ===============================================
    #           - tokenizer evaluation -          
    # ===============================================

    def compute_reconstruction_error(self, raw_traj):
        raw_traj = raw_traj.to(self.device, dtype=torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        tokens, _ = self.encode(raw_traj)
        reconstruct_trajs = self.reconstruct_traj(tokens)
        error = torch.mean((raw_traj - reconstruct_trajs) ** 2)
        return error

    @autocast_float32
    def visualize_reconstruction_error(self, raw_traj, max_vis_samples=3, update_bounds=True):
        raw_traj = raw_traj.to(self.device, dtype=torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        tokens, params_dict = self.encode(raw_traj, update_bounds=update_bounds)
        pos = self.reconstruct_traj(tokens)
        pos = pos.detach().cpu().numpy()
        raw_traj = raw_traj.detach().cpu().numpy()
        x_vals = np.linspace(0, self.duration, raw_traj.shape[1])

        batch_size, time_steps, dof = raw_traj.shape
        # Plot both generated and ground truth sine waves
        
        if batch_size > max_vis_samples:
            batch_size = max_vis_samples
        
        for sample_idx in range(batch_size):
            fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

            for i in range(dof):
                axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                color='b')
                axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                color='r')
                axes[i].set_ylabel(f"DOF {i + 1}")
                axes[i].grid(True)
                axes[i].legend(loc="best")

            axes[-1].set_xlabel("Timesteps")
            plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    @autocast_float32
    def visualize_reconstruction_error_with_llm_tokenizer(self, raw_traj,
                                                          save_path=None):
        raw_traj = raw_traj.to(self.device, dtype=torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        tokens, params_dict = self.encode(raw_traj, update_bounds=True)
        llm_tokens = self.tokens_to_llm_tokens(tokens)
        # reconstruct the trajectory from the llm tokens
        pos = self.reconstruct_from_llm_tokens(llm_tokens)
        pos = pos.detach().cpu().numpy()
        raw_traj = raw_traj.detach().cpu().numpy()
        x_vals = np.linspace(0, self.duration, raw_traj.shape[1])

        batch_size, time_steps, dof = raw_traj.shape
        # Plot both generated and ground truth sine waves
        for sample_idx in range(batch_size):
            fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

            for i in range(dof):
                axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                color='b')
                axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                color='r')
                axes[i].set_ylabel(f"DOF {i + 1}")
                axes[i].grid(True)
                axes[i].legend(loc="best")

            axes[-1].set_xlabel("Timesteps")
            plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save the figure with the specified naming format
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                save_filename = os.path.join(save_path, f"sample_{sample_idx}.png")
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            # Close the figure to free memory (important when processing many plots)
            plt.show()
            plt.close(fig)
    
    @autocast_float32
    def visualize_reconstruction_error_with_cont_tokenizer(self, raw_traj,
                                                          save_path=None):
        raw_traj = raw_traj.to(self.device, dtype=torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        continous_tokens, _ = self.encode_continuous(raw_traj, update_bounds=True)
        # reconstruct the trajectory from the llm tokens
        pos = self.reconstruct_traj_continuous(continous_tokens)
        pos = pos.detach().cpu().numpy()
        raw_traj = raw_traj.detach().cpu().numpy()
        x_vals = np.linspace(0, self.duration, raw_traj.shape[1])

        batch_size, time_steps, dof = raw_traj.shape
        # Plot both generated and ground truth sine waves
        for sample_idx in range(batch_size):
            fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)

            for i in range(dof):
                axes[i].plot(x_vals, pos[sample_idx, :, i], marker='o', label='reconstruct', linestyle='-',
                                color='b')
                axes[i].plot(x_vals, raw_traj[sample_idx, :, i], marker='*', label='ground_truth', linestyle='--',
                                color='r')
                axes[i].set_ylabel(f"DOF {i + 1}")
                axes[i].grid(True)
                axes[i].legend(loc="best")

            axes[-1].set_xlabel("Timesteps")
            plt.suptitle(f"Visualization of Sample {sample_idx} in Batch")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save the figure with the specified naming format
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                save_filename = os.path.join(save_path, f"sample_{sample_idx}.png")
                plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            # Close the figure to free memory (important when processing many plots)
            plt.show()
            # plt.close(fig)

