"""Load the robotics data pipeline and print one batch per dataloader.

Fails fast (non-zero exit) when the config, dataset roots or transforms are
broken, so it can be a launcher stage before long tokenizer runs:

    PYTHONPATH=.:MP_lite_PyTorch python train/check_data.py --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check the tokenizer data pipeline.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true", help="Only pull one train batch.")
    args = parser.parse_args()

    t0 = time.time()
    from train.data import prepare_dataloaders  # heavy import

    print(f"[check_data] import ok in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    example_actions, dl_train, dl_evals = prepare_dataloaders(args.batch_size, num_workers=args.num_workers)
    print(
        f"[check_data] datasets ready in {time.time() - t0:.1f}s; example actions {tuple(example_actions.shape)}; "
        f"eval datasets ({len(dl_evals)}): {list(dl_evals)}",
        flush=True,
    )
    t0 = time.time()
    batch = next(iter(dl_train))
    actions = batch["actions"]
    print(
        f"[check_data] first train batch in {time.time() - t0:.1f}s: actions {tuple(actions.shape)} dtype={actions.dtype} "
        f"keys={sorted(batch.keys())[:12]}",
        flush=True,
    )
    if actions.dim() != 3:
        print(f"[check_data] unexpected action rank {actions.dim()}", flush=True)
        return 2
    per_dim_max = actions.abs().amax(dim=(0, 1))
    print("[check_data] per-dim abs max:", [round(float(v), 3) for v in per_dim_max], flush=True)
    print("[check_data] all-zero dims:", [i for i, v in enumerate(per_dim_max.tolist()) if v == 0.0], flush=True)
    if not args.skip_eval:
        for name, loader in dl_evals.items():
            t0 = time.time()
            eval_batch = next(iter(loader))
            print(f"[check_data] eval {name}: actions {tuple(eval_batch['actions'].shape)} in {time.time() - t0:.1f}s", flush=True)
    print("[check_data] DATA OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
