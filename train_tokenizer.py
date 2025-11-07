import os

os.environ["OPENPI_DATA_HOME"] = (
    "/mnt/virtual_ai0001071-01239_SR006-nfs2/apanasevich/openpi/assets"
)
os.environ["HF_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache"


from lerobot.common.datasets.create_dataloader import create_lerobot_dataloader
from lerobot.common.datasets.data_config import (
    LeRobotAgibotTwoFingerDataConfig,
    LeRobotAgibotDexHandDataConfig,
)
from rich_argparse import RichHelpFormatter
import argparse

import torch

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
from omegaconf import OmegaConf
from hydra.utils import instantiate
from lerobot.common.datasets.create_dataloader import create_lerobot_dataset_by_config
from torch.utils.data import DataLoader


def get_data(
    dataset_config_path: str, 
    assets_dir: str, 
    action_horizon: int | None = None,
    action_dim: int | None = None
):
    cfg = OmegaConf.load(dataset_config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if action_horizon is not None:
        cfg['action_horizon'] = action_horizon
    
    data_config = instantiate(cfg)
    model_cfg = PI0Config()
    lerobot_dataset = create_lerobot_dataset_by_config(
        data_config_factory=data_config,
        model_config=model_cfg,
        assets_dirs=assets_dir,
        normalization_mode="mean_std", #it does not matter
        skip_norm_stats=True,
        skip_model_transforms=True,
        return_norm_stats=False,
        )
    
    class InnerDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, action_horizon, action_dim):
            self.dataset = dataset
            self.action_horizon = action_horizon
            self.action_dim = action_dim
            
            if self.action_dim is None:
                self.action_dim = self.dataset[0]['actions'].shape[1]

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.dataset[idx]['actions'][:self.action_horizon, :self.action_dim]

    return InnerDataset(lerobot_dataset, action_horizon, action_dim)

