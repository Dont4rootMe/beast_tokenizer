import torch


def continuous_to_discrete(tensor, min_val=None, max_val=None, num_bins=256):
    """Convert a continuous tensor to discrete tokens."""

    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()

    scale = torch.clamp(max_val - min_val, min=1e-8)
    normalized_tensor = (tensor - min_val) / scale
    normalized_tensor = torch.clamp(normalized_tensor, 0, 1)

    discrete_tensor = torch.round(normalized_tensor * (num_bins - 1)).to(torch.long)
    return discrete_tensor


def discrete_to_continuous(discrete_tensor, min_val=0, max_val=1, num_bins=256):
    """Map discrete tokens back to the original continuous range."""

    normalized_tensor = discrete_tensor.float() / (num_bins - 1)
    continuous_tensor = normalized_tensor * (max_val - min_val) + min_val
    continuous_tensor = torch.clamp(continuous_tensor, min_val, max_val)
    return continuous_tensor


def normalize_tensor(tensor, w_min, w_max, norm_min=-1.0, norm_max=1.0):
    """Normalize a tensor from [w_min, w_max] to [norm_min, norm_max]."""

    clipped_tensor = torch.clamp(tensor, w_min, w_max)
    normalized = (clipped_tensor - w_min) / torch.clamp(w_max - w_min, min=1e-8)
    normalized = normalized * (norm_max - norm_min) + norm_min
    return normalized


def denormalize_tensor(normalized_tensor, w_min, w_max, norm_min=-1.0, norm_max=1.0):
    """Map a normalized tensor back to [w_min, w_max]."""

    clipped_tensor = torch.clamp(normalized_tensor, norm_min, norm_max)
    denormalized = (clipped_tensor - norm_min) / torch.clamp(norm_max - norm_min, min=1e-8)
    denormalized = denormalized * (w_max - w_min) + w_min
    return denormalized
