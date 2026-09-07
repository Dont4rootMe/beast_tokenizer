"""Resolve dataset-config paths stored inside ``config.ckpt``.

``config.ckpt`` embeds absolute ``${_load_config:/.../lerobot/conf/...yaml}`` paths
that point at a specific lerobot-fork checkout. When that checkout moves on and a
referenced YAML disappears, set ``BEAST_LEROBOT_ROOT`` to another checkout (for
example a snapshot pinned to an older commit): the part of the path after the
first ``lerobot/conf/`` marker is re-rooted there.
"""
from __future__ import annotations

import os
from typing import Optional

ENV_VAR = "BEAST_LEROBOT_ROOT"
BASE_VLM_ENV_VAR = "BEAST_BASE_VLM_MODEL"
_MARKER = "/lerobot/conf/"


def resolve_config_path(rel_path: str, alt_root: Optional[str] = None) -> str:
    """Return the path to load for ``rel_path``.

    Relative paths are joined with the current working directory; absolute paths are
    kept. If ``alt_root`` (default: the ``BEAST_LEROBOT_ROOT`` environment variable)
    is set, the ``lerobot/conf/...`` suffix is re-rooted under ``alt_root`` whenever
    that file exists, so every dataset config comes from one consistent checkout.
    Otherwise the original path is returned unchanged.
    """
    path = os.path.join(os.getcwd(), rel_path)
    if alt_root is None:
        alt_root = os.environ.get(ENV_VAR)
    if alt_root:
        idx = path.find(_MARKER)
        if idx >= 0:
            candidate = os.path.join(alt_root, path[idx + 1:])
            if os.path.exists(candidate):
                return candidate
    return path


def resolve_base_vlm_model(default: str) -> str:
    """Return the VLM tokenizer source: ``$BEAST_BASE_VLM_MODEL`` (a local directory,
    e.g. a Hugging Face cache snapshot) when set, otherwise ``default`` (a repo id).

    Loading by repo id needs network access to the Hub even for cached, gated
    models with recent ``transformers`` versions; a local directory does not.
    """
    override = os.environ.get(BASE_VLM_ENV_VAR)
    if not override:
        return default
    if not os.path.isdir(override):
        raise FileNotFoundError(f"{BASE_VLM_ENV_VAR}={override!r} is not a directory")
    return override
