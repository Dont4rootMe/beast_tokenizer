import os
import json

import optuna

os.environ["OPENPI_DATA_HOME"] = (
    "/mnt/virtual_ai0001071-01239_SR006-nfs2/apanasevich/openpi/assets"
)
os.environ["HF_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache"

from lerobot.common.datasets.torch_transforms import compose
from lerobot.common.datasets.create_dataloader import create_lerobot_dataloader
from lerobot.common.datasets.data_config import (
    LeRobotAgibotTwoFingerDataConfig,
    LeRobotAgibotDexHandDataConfig,
)
from matplotlib import pyplot as plt
import seaborn as sns

from beast.bspline_tokenizer import BSpline_Tokenizer
import numpy as np

import hydra
from rich_argparse import RichHelpFormatter
from accelerate.utils import broadcast_object_list
import argparse

import torch
from lerobot.common.utils.inference_transforms import get_torch_input_transforms, get_torch_output_transforms
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from lerobot.common.datasets.data_config import (
    AssetsConfig as LeRobotAssetsConfig
)
from lerobot.common.datasets.data_config import DataConfig as LeRobotBaseDataConfig
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from tqdm import tqdm
from accelerate import Accelerator
from lerobot.common.utils.normalize import RunningStats, save as save_stats
import torch.distributed as dist
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from hydra.utils import instantiate
from lerobot.common.datasets.create_dataloader import create_lerobot_dataset_by_config
from torch.utils.data import DataLoader

from typing import Any
import random
import logging

from functools import partial


def create_train_val_datasets_distributed(
    data_config_factory: Any,
    model_config: Any,
    assets_dirs: str,
    accelerator: Accelerator,  # Add accelerator parameter
    val_split: float = 0.1,
    seed: int = 42,
    map_to_unified_space: bool = False,
    use_validation_list: bool = False,
    recompute_norm_stats: bool = False,
    **dataset_kwargs
):
    """
    Create train and validation datasets with distributed-aware episode splitting.
    Only main process loads metadata, then broadcasts splits to all processes.
    
    Logic flow:
    1. First, check if validation episodes are provided in data config via 'validation_episodes' key
    2. If validation episodes files are found, load them and create val_episodes_dict from them
    3. If no validation episodes files are found, do random shuffling and splitting based on val_split ratio
    4. Train episodes are always created as the complement of validation episodes
    
    The validation_episodes JSON file can contain:
    - Direct list: [0, 1, 2, ...] (applies to all datasets)
    - Multi-dataset dict: {dataset_name: [0, 1, 2, ...]}
    """
    class DummyFactory:
        def __init__(self, cfg_item):
            self.cfg_item = cfg_item
        
        def create(self, *args, **kwargs):
            return self.cfg_item
    
    # Get data config to determine dataset type (needed for output pipeline creation)
    data_cfg = data_config_factory.create(assets_dirs, model_config)
    
    # Check if we have mixture configs
    if hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs:
        # Multi-dataset scenario
        dataset_configs = data_cfg.mixture_configs
        dataset_names = [cfg_item.repo_id for cfg_item in dataset_configs] 
        for i, cfg_item in enumerate(dataset_configs):
            cfg_name = dataset_names[i]
    else:
        # Single dataset scenario
        dataset_configs = [data_cfg]
        dataset_names = ["main"]
    
    # ONLY MAIN PROCESS loads metadata and creates splits
    if accelerator.is_main_process:
        train_episodes_dict = {}
        val_episodes_dict = {}
        # First, try to load validation episodes from data config
        validation_episodes_loaded = False
        
        for i, cfg_item in enumerate(dataset_configs):
            cfg_name = dataset_names[i]
            
            # Check if validation episodes are provided for this dataset
            if use_validation_list and hasattr(cfg_item, 'validation_episodes') and cfg_item.validation_episodes:
                validation_episodes_path = cfg_item.validation_episodes
                if os.path.exists(validation_episodes_path):
                    try:
                        with open(validation_episodes_path, 'r') as f:
                            validation_episodes_data = json.load(f)
                        
                        # Handle different JSON structures for this specific dataset
                        if isinstance(validation_episodes_data, list):
                            # Direct list format: [0, 1, 2, ...]
                            val_episodes_dict[cfg_name] = validation_episodes_data
                        elif isinstance(validation_episodes_data, dict):
                            # Multi-dataset format: {dataset_name: [episodes]}
                            if cfg_name in validation_episodes_data:
                                val_episodes_dict[cfg_name] = validation_episodes_data[cfg_name]
                            else:
                                # Fallback to random splitting for this dataset
                                val_episodes_dict[cfg_name] = None
                        else:
                            # Fallback to random splitting for this dataset
                            val_episodes_dict[cfg_name] = None
                        
                        if val_episodes_dict[cfg_name] is not None:
                            validation_episodes_loaded = True
                            logging.info(f"Loaded validation episodes for {cfg_name} from data config {validation_episodes_path}: {val_episodes_dict[cfg_name]}")
                        else:
                            logging.warning(f"Could not extract validation episodes for {cfg_name} from {validation_episodes_path}")
                    except Exception as e:
                        logging.warning(f"Failed to load validation episodes for {cfg_name} from {validation_episodes_path}: {e}")
                        val_episodes_dict[cfg_name] = None
                else:
                    logging.warning(f"Validation episodes file {validation_episodes_path} not found for {cfg_name}")
                    val_episodes_dict[cfg_name] = None
            else:
                val_episodes_dict[cfg_name] = None
        
        # If no validation episodes were loaded, do random splitting
        if not validation_episodes_loaded:
            logging.warning("No validation episodes files found or use_validation_list is False, using random splitting")
            for i, cfg_item in enumerate(dataset_configs):
                cfg_name = dataset_names[i]
                
                # Load metadata for this dataset
                info_path = Path(cfg_item.root_dir) / "meta" / "info.json"
                with open(info_path, "r") as f:
                    total_episodes = json.load(f)["total_episodes"]
                
                # Create random episode splits
                all_episodes = list(range(total_episodes))
                random.seed(seed + hash(cfg_name))
                random.shuffle(all_episodes)
                
                n_val = max(1, int(total_episodes * val_split))
                val_episodes = all_episodes[:n_val]
                train_episodes = all_episodes[n_val:]
                
                val_episodes_dict[cfg_name] = val_episodes
                train_episodes_dict[cfg_name] = train_episodes
        else:
            # Create train episodes as complement of validation episodes
            for i, cfg_item in enumerate(dataset_configs):
                cfg_name = dataset_names[i]
                
                # Load metadata for this dataset
                info_path = Path(cfg_item.root_dir) / "meta" / "info.json"
                with open(info_path, "r") as f:
                    total_episodes = json.load(f)["total_episodes"]
                
                if val_episodes_dict[cfg_name] is not None:
                    # Create train episodes as complement of validation episodes
                    all_episodes = set(range(total_episodes))
                    val_episodes_set = set(val_episodes_dict[cfg_name])
                    train_episodes = sorted(list(all_episodes - val_episodes_set))
                    train_episodes_dict[cfg_name] = train_episodes
                else:
                    # Fallback to random splitting for this dataset
                    all_episodes = list(range(total_episodes))
                    random.seed(seed + hash(cfg_name))
                    random.shuffle(all_episodes)
                    
                    n_val = max(1, int(total_episodes * val_split))
                    val_episodes = all_episodes[:n_val]
                    train_episodes = all_episodes[n_val:]
                    
                    val_episodes_dict[cfg_name] = val_episodes
                    train_episodes_dict[cfg_name] = train_episodes
    else:
        # Other processes wait for broadcast
        train_episodes_dict = None
        val_episodes_dict = None
    
    # BROADCAST episode splits from main process to all processes
    train_episodes_dict = broadcast_object_list([train_episodes_dict])[0]
    val_episodes_dict = broadcast_object_list([val_episodes_dict])[0]
    
    # Check if we have mixture configs (data_cfg already created above)
    is_mixture_cfg = hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs

    # If a dataset provides an episodes allowlist file, intersect BEFORE sharding to avoid empty subsets
    cfg_map = {}
    if is_mixture_cfg:
        for item in data_cfg.mixture_configs:
            cfg_map[item.repo_id] = item
    else:
        cfg_map["main"] = data_cfg

    allowed_by_cfg = {}
    for name, cfg_item in cfg_map.items():
        allowlist_path = getattr(cfg_item, 'episodes_list_file', None)
        allowed_set = None
        try:
            if allowlist_path and os.path.exists(allowlist_path):
                with open(allowlist_path, 'r') as f:
                    allowed_set = set(json.load(f))
        except Exception:
            allowed_set = None
        allowed_by_cfg[name] = allowed_set

    def _apply_allowlist(eps_dict: dict[str, list[int]]):
        out = {}
        for name, eps in eps_dict.items():
            allowed = allowed_by_cfg.get(name)
            if isinstance(eps, list) and allowed is not None:
                filtered = [e for e in eps if e in allowed]
                if len(filtered) == 0 and len(allowed) > 0:
                    # Fallback to all allowed episodes to avoid empty datasets
                    filtered = sorted(list(allowed))
                elif len(allowed) == 0:
                    raise Exception(f'The validation episodes are empty for the dataset: {name}. Please check the episodes list file: {allowlist_path} or the validation episodes list: meta/validation_episodes.json')
                out[name] = filtered
            else:
                out[name] = eps
        return out

    train_episodes_dict = _apply_allowlist(train_episodes_dict)
    val_episodes_dict = _apply_allowlist(val_episodes_dict)

    # Partition validation episodes across processes so each worker evaluates on a different subset
    # Keep keys identical across ranks to avoid collective mismatches during metric aggregation
    if hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs:
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        val_episodes_dict_local = {}
        for cfg_name, eps in val_episodes_dict.items():
            if isinstance(eps, list):
                if len(eps) >= world_size:
                    # Round-robin sharding
                    shard = eps[rank::world_size]
                elif len(eps) > 0:
                    # Too few episodes for strict sharding; assign at least one per rank via wrap-around
                    shard = [eps[rank % len(eps)]]
                else:
                    shard = []
                val_episodes_dict_local[cfg_name] = shard
            else:
                # Fallback: if episodes are not a list, keep as-is
                val_episodes_dict_local[cfg_name] = eps
        
        
        # Shard training episodes across processes with safe fallback for tiny splits
        train_episodes_dict_local = {}
        for cfg_name, eps in train_episodes_dict.items():
            if isinstance(eps, list):
                if len(eps) >= world_size:
                    shard = eps[rank::world_size]
                elif len(eps) > 0:
                    shard = [eps[rank % len(eps)]]
                else:
                    shard = []
                train_episodes_dict_local[cfg_name] = shard
            else:
                train_episodes_dict_local[cfg_name] = eps
    else:
        train_episodes_dict_local = train_episodes_dict
        val_episodes_dict_local = val_episodes_dict


    # ALL PROCESSES create datasets using their local episode splits
    train_dataset, norm_stats = create_lerobot_dataset_by_config(
        data_config_factory=data_config_factory,
        model_config=model_config,
        assets_dirs=assets_dirs,
        episodes=train_episodes_dict_local,
        normalization_mode=model_config.normalization_mode,
        return_norm_stats=True,
        map_to_unified_space=map_to_unified_space,
        recompute_norm_stats=recompute_norm_stats,
        **dataset_kwargs
    )
    # If mixture dataset, set per-rank RNG to reduce cross-rank duplication
    if hasattr(train_dataset, "set_rng"):
        try:
            import numpy as _np
            train_dataset.set_rng(_np.random.RandomState(seed + rank))
        except Exception:
            pass
    
    val_datasets_dict = {}
    # data_cfg already created above, reuse it
    is_mixture = hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs
    
    # Initialize output_pipeline_dict for all processes
    output_pipeline_dict = {}
    
    for cfg_name, eps in val_episodes_dict_local.items():
        if is_mixture:
            cfg_item = next(item for item in data_cfg.mixture_configs if item.repo_id == cfg_name)
        else:
            cfg_item = data_cfg
        norm_stats_item = norm_stats[cfg_item.repo_id]
        factory = DummyFactory(cfg_item)
        val_dataset = create_lerobot_dataset_by_config(
            data_config_factory=factory,
            model_config=model_config,
            assets_dirs=assets_dirs,
            episodes=eps,
            normalization_mode=model_config.normalization_mode,
            return_norm_stats=False,
            recompute_norm_stats=False,
            precomputed_norm_stats=norm_stats,
            map_to_unified_space=map_to_unified_space,
            **dataset_kwargs
        )
        val_datasets_dict[cfg_name] = val_dataset
        output_pipeline_dict[cfg_name] = compose(
            get_torch_output_transforms(
                norm_stats=norm_stats_item,
                policy_config=model_config,
                data_config_factory=DummyFactory(cfg_item),
                assets_dirs=assets_dirs,
                normalization_mode=model_config.normalization_mode,
                map_to_unified_space=map_to_unified_space
        ))

    return train_dataset, val_datasets_dict, norm_stats, output_pipeline_dict

def instantiate_data_config(cfg: DictConfig, add_kwargs: dict = None):
    """Instantiate robotics dataset config.

    - For mixture configs: mutate each nested data_config only for keys that already exist
    (e.g., map_to_unified_space, map_to_humanoid), then instantiate with _recursive_=False.
    - For individual configs: set keys on the config if they already exist, then instantiate with _recursive_=True.
    """
    try:
        is_mixture = cfg._target_.split(".")[-1] == "MixtureDataConfigFactory"
    except Exception:
        is_mixture = False

    if is_mixture and hasattr(cfg, "datasets_with_weights") and cfg.datasets_with_weights is not None:
        assert cfg.data_configs is None, "both datasets_with_weights and data_configs are set"
        assert cfg.weights is None, "both datasets_with_weights and weights are set"
        datasets_list = [ds_cfg.path for ds_cfg in cfg.datasets_with_weights]
        weights_list = [ds_cfg.weight for ds_cfg in cfg.datasets_with_weights]
        cfg.data_configs = datasets_list
        cfg.weights = weights_list
        del cfg.datasets_with_weights

    if add_kwargs:
        if is_mixture and hasattr(cfg, "data_configs") and cfg.data_configs is not None:
            # Update only existing keys in each nested dataset cfg to avoid unknown-arg errors
            for idx in range(len(cfg.data_configs)):
                dc = cfg.data_configs[idx]
                try:
                    # DictConfig supports 'in' and item assignment
                    for k, v in add_kwargs.items():
                        if isinstance(dc, DictConfig):
                            if k in dc:
                                dc[k] = v
                        elif isinstance(dc, dict):
                            if k in dc:
                                dc[k] = v
                except Exception:
                    # Best-effort; skip problematic entries
                    pass
        else:
            # Individual dataset config: only set keys that exist in cfg
            for k, v in add_kwargs.items():
                try:
                    if k in cfg:
                        cfg[k] = v
                except Exception:
                    pass

    if is_mixture:
        return hydra.utils.instantiate(cfg, _recursive_=False)
    else:
        return hydra.utils.instantiate(cfg, _recursive_=True)

def get_datasets():
    try:
        OmegaConf.register_resolver(
            "_load_config", lambda rel_path: OmegaConf.load(os.path.join(os.getcwd(), rel_path))
        )
    except:
        pass

    accelerator = Accelerator()
    assets_dir = '/mnt/virtual_ai0001071-01239_SR006-nfs2/apanasevich/pi0_assets_v4'

    cfg = torch.load('config.ckpt', weights_only=False)
    # cfg.robotics_dataset.data_configs = cfg.robotics_dataset.data_configs[:2]
    # cfg.robotics_dataset.weights = cfg.robotics_dataset.weights[:2]

    map_to_unified_space = False
    map_to_humanoid = False
    add_kwargs = {
        'map_to_unified_space': map_to_unified_space,
        'map_to_humanoid': map_to_humanoid,
    }
    robotics_dataset_factory = instantiate_data_config(cfg.robotics_dataset, add_kwargs)
    policy_config = hydra.utils.instantiate(cfg.policy.policy_config)

    val_split = 0.02

    robotics_dataset, val_datasets_dict, norm_stats, output_pipeline_dict = create_train_val_datasets_distributed(
        data_config_factory=robotics_dataset_factory,
        model_config=policy_config,
        assets_dirs=assets_dir,
        accelerator=accelerator,
        val_split=val_split,
        seed=42,
        map_to_unified_space=map_to_unified_space,
        use_validation_list=True,
        recompute_norm_stats=False,
        
    )
    
    return robotics_dataset, val_datasets_dict, norm_stats, output_pipeline_dict


def main():
    robotics_dataset, val_datasets_dict, norm_stats, output_pipeline_dict = get_datasets()
    dtl = DataLoader(robotics_dataset, batch_size=32, shuffle=True)
    for dataset in dtl.dataset._datasets:
        dataset._dataset._dataset.return_fake_images = True

    actions_len, actions_dof = robotics_dataset[0]['actions'].shape
    
    tokenizer = BSpline_Tokenizer(
        num_basis=32,
        vocab_size=800,
        degree_p=1,
        
        num_dof=actions_dof,
        seq_len=actions_len,
        gripper_indices=[],
        gripper_zero_order=False,
        init_pos=False,
        device='cpu'
    )
    tokenizer.fit_parameters(dtl, max_samples=100_000)
    tokenizer.save_pretrained('big_train_tokenizers/32_800_1')
    
    errors_l2 = []
    errors_l1 = []
    for batch in tqdm(dtl, total=12_500, desc="Computing reconstruction errors"):
        if len(errors_l2) >= 12_500:
            break
        actions = batch['actions']
        error_l2, error_l1 = tokenizer.compute_reconstruction_error(actions)
        error_l2, error_l1 = error_l2.item(), error_l1.item()
        errors_l2.append(error_l2)
        errors_l1.append(error_l1)
    
    with open('big_train_tokenizers/50_1000_0/errors.json', 'w') as f:
        json.dump({
            'errors_l2': errors_l2,
            'errors_l1': errors_l1,
        }, f)
    
    with open('big_train_tokenizers/50_1000_0/stats.txt', 'w') as f:
        print('Mean reconstruction error l2:', np.mean(errors_l2), file=f)
        print('Std reconstruction error l2:', np.std(errors_l2), file=f)
        print('Max reconstruction error l2:', np.max(errors_l2), file=f)
        print('Min reconstruction error l2:', np.min(errors_l2), file=f)
        print('', file=f)
        print('Mean reconstruction error l1:', np.mean(errors_l1), file=f)
        print('Std reconstruction error l1:', np.std(errors_l1), file=f)
        print('Max reconstruction error l1:', np.max(errors_l1), file=f)
        print('Min reconstruction error l1:', np.min(errors_l1), file=f)
        
    sns.histplot(errors_l2, bins=100, alpha=0.5, color='b', kde=True)
    plt.savefig('big_train_tokenizers/50_1000_0/histogram_l2.png')
    plt.close()
    
    sns.histplot(errors_l1, bins=100, alpha=0.5, color='b', kde=True)
    plt.savefig('big_train_tokenizers/50_1000_0/histogram_l1.png')
    plt.close()
    
    actions = next(iter(dtl))['actions'][:5]
    tokenizer.visualize_reconstruction_error(actions, save_path='big_train_tokenizers/50_1000_0')

if __name__ == '__main__':
    main()