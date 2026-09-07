"""Tests for train.config_paths (pure Python, no torch needed).

Run: python train/test_config_paths.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.config_paths import dataset_overrides_from_env, resolve_base_vlm_model, resolve_config_path  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    # Isolate from the caller's environment: the launcher exports these for real runs.
    saved_env = {k: os.environ.pop(k, None) for k in ("BEAST_LEROBOT_ROOT", "BEAST_BASE_VLM_MODEL", "BEAST_ACTION_HORIZON")}
    try:
        return _run_tests()
    finally:
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v


def _run_tests() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        original_root = tmp / "constant_repos" / "lerobot-fork"
        alt_root = tmp / "snapshot"
        rel = Path("lerobot/conf/robotics_dataset/individual/rdt.yaml")
        (alt_root / rel).parent.mkdir(parents=True)
        (alt_root / rel).write_text("x: 1\n")

        # 1. Existing absolute path is returned unchanged.
        existing = original_root / "lerobot/conf/robotics_dataset/individual/action_net.yaml"
        existing.parent.mkdir(parents=True)
        existing.write_text("y: 2\n")
        check(resolve_config_path(str(existing), alt_root=None) == str(existing), "existing path must be returned as is")

        # 2. Missing path under the original root is remapped to alt_root by the lerobot/conf suffix.
        missing = original_root / rel
        got = resolve_config_path(str(missing), alt_root=str(alt_root))
        check(got == str(alt_root / rel), f"expected remap to {alt_root / rel}, got {got}")

        # 2b. When alt_root is set, an existing original is still re-rooted if the alt file exists
        #     (all configs must come from one consistent checkout).
        (alt_root / "lerobot/conf/robotics_dataset/individual/action_net.yaml").write_text("z: 3\n")
        got = resolve_config_path(str(existing), alt_root=str(alt_root))
        check(got == str(alt_root / "lerobot/conf/robotics_dataset/individual/action_net.yaml"), f"alt_root must win over the original when both exist, got {got}")
        # 2c. ...but falls back to the original when the alt file is absent.
        other = original_root / "lerobot/conf/robotics_dataset/individual/only_here.yaml"
        other.write_text("w: 4\n")
        check(resolve_config_path(str(other), alt_root=str(alt_root)) == str(other), "missing alt file -> original")

        # 3. Missing path without alt_root (or without a lerobot/conf marker) is returned unchanged.
        check(resolve_config_path(str(missing), alt_root=None) == str(missing), "no alt_root -> unchanged")
        odd = tmp / "elsewhere" / "file.yaml"
        check(resolve_config_path(str(odd), alt_root=str(alt_root)) == str(odd), "no marker -> unchanged")

        # 4. Relative paths are resolved against cwd first.
        cwd = os.getcwd()
        try:
            os.chdir(alt_root)
            got_rel = resolve_config_path(str(rel), alt_root=None)
            check(os.path.realpath(got_rel) == os.path.realpath(alt_root / rel), "relative path joins cwd")
        finally:
            os.chdir(cwd)

        # 5. alt_root defaults to the BEAST_LEROBOT_ROOT environment variable.
        old = os.environ.get("BEAST_LEROBOT_ROOT")
        os.environ["BEAST_LEROBOT_ROOT"] = str(alt_root)
        try:
            check(resolve_config_path(str(missing)) == str(alt_root / rel), "env var must supply alt_root")
        finally:
            if old is None:
                del os.environ["BEAST_LEROBOT_ROOT"]
            else:
                os.environ["BEAST_LEROBOT_ROOT"] = old
    # 6. resolve_base_vlm_model: env override wins, otherwise the default repo id is kept.
    old = os.environ.pop("BEAST_BASE_VLM_MODEL", None)
    try:
        check(resolve_base_vlm_model("google/paligemma-3b-pt-224") == "google/paligemma-3b-pt-224", "no override -> default")
        with tempfile.TemporaryDirectory() as snap:
            os.environ["BEAST_BASE_VLM_MODEL"] = snap
            check(resolve_base_vlm_model("google/paligemma-3b-pt-224") == snap, "existing local dir override must win")
        os.environ["BEAST_BASE_VLM_MODEL"] = "/definitely/missing/dir"
        try:
            resolve_base_vlm_model("google/paligemma-3b-pt-224")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing override dir must raise FileNotFoundError")
    finally:
        if old is None:
            os.environ.pop("BEAST_BASE_VLM_MODEL", None)
        else:
            os.environ["BEAST_BASE_VLM_MODEL"] = old
    # 7. dataset_overrides_from_env: BEAST_ACTION_HORIZON -> {"action_horizon": int}; invalid values raise.
    saved = os.environ.pop("BEAST_ACTION_HORIZON", None)
    try:
        check(dataset_overrides_from_env() == {}, "no env -> no overrides")
        os.environ["BEAST_ACTION_HORIZON"] = "10"
        check(dataset_overrides_from_env() == {"action_horizon": 10}, "horizon override must parse to int")
        os.environ["BEAST_ACTION_HORIZON"] = "0"
        try:
            dataset_overrides_from_env()
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive horizon must raise ValueError")
    finally:
        if saved is None:
            os.environ.pop("BEAST_ACTION_HORIZON", None)
        else:
            os.environ["BEAST_ACTION_HORIZON"] = saved
    print("CONFIG PATHS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
