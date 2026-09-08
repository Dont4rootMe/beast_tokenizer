"""Collate helpers that tolerate heterogeneous sample dicts.

Mixture datasets do not all carry the same keys (for example only some have a
``subtask`` annotation); ``torch.utils.data.default_collate`` takes the key set
of the first sample and raises ``KeyError`` on the others. The tokenizer only
needs a few tensors, so collate exactly those.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch


def select_keys_collate(keys: Sequence[str], optional: Sequence[str] = ()) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Return a collate function that stacks ``keys`` (required) and ``optional`` keys.

    Required keys must exist in every sample and are stacked into tensors.
    Optional keys are included only when present in every sample; tensor-like
    values are stacked, anything else is returned as a list.
    """
    required = tuple(keys)
    optional_keys = tuple(optional)

    def _stack(values: List[Any]) -> Any:
        try:
            return torch.stack([torch.as_tensor(v) for v in values])
        except (TypeError, ValueError, RuntimeError):
            return values

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in required:
            try:
                values = [sample[key] for sample in batch]
            except KeyError as exc:
                available = sorted(batch[0].keys()) if batch else []
                raise KeyError(f"required key {key!r} missing in a sample; first sample keys: {available[:20]}") from exc
            out[key] = torch.stack([torch.as_tensor(v) for v in values])
        for key in optional_keys:
            if batch and all(key in sample for sample in batch):
                out[key] = _stack([sample[key] for sample in batch])
        return out

    return _collate
