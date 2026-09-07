"""Sweep BEAST (num_basis, degree) configurations on cached action batches.

Example (from the repository root):

    PYTHONPATH=.:MP_lite_PyTorch python train/sweep_beast.py \
        --num-basis-grid 4,5,6,8 --degree-grid 2,3 \
        --fit-beast-max-samples 1000 --max-eval-samples 300 --out-dir sweep_results

Batches are read from the dataloaders once and cached in memory so every
configuration sees identical data. After each configuration the summary files
``summary.json`` / ``summary.csv`` are rewritten, so a killed run keeps its rows
and a restart skips rows that already exist.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer
from train.eval import evaluate_tokenizer

COLUMNS = [
    "config", "num_basis", "degree", "vocab_size", "num_dof", "dataset", "is_bpe",
    "tokens_pre_bpe", "mean_tokens", "std_tokens", "max_tokens",
    "mean_l2", "mean_l1", "mean_max_abs", "max_abs_err", "num_chunks",
]


def _parse_grid(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _cache_batches(loader: Iterable[dict], max_batches: Optional[int], key: str = "actions") -> List[dict]:
    cached: List[dict] = []
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        cached.append({key: batch[key].detach().to("cpu").clone()})
    return cached


def _write_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in COLUMNS})


def _row(config: str, nb: int, deg: int, vocab: int, num_dof: int, dataset: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": config,
        "num_basis": nb,
        "degree": deg,
        "vocab_size": vocab,
        "num_dof": num_dof,
        "dataset": dataset,
        "is_bpe": bool(stats["is_bpe"]),
        "tokens_pre_bpe": stats["tokens_pre_bpe"],
        "mean_tokens": stats["mean_tokens"],
        "std_tokens": stats["std_tokens"],
        "max_tokens": stats["max_tokens"],
        "mean_l2": stats["mean_l2"],
        "mean_l1": stats["mean_l1"],
        "mean_max_abs": stats["mean_max_abs"],
        "max_abs_err": stats["max_max_abs"],
        "num_chunks": stats["num_chunks"],
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'config':<12} {'bpe':<5} {'dataset':<28} {'tok_pre':>7} {'tok_mean':>9} {'mean_l1':>10} {'mean_l2':>10} {'max_abs':>9}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["num_basis"], r["degree"], r["is_bpe"], r["dataset"])):
        print(
            f"{r['config']:<12} {str(r['is_bpe']):<5} {r['dataset'][:28]:<28} {r['tokens_pre_bpe']:>7} "
            f"{r['mean_tokens']:>9.1f} {r['mean_l1']:>10.5f} {r['mean_l2']:>10.6f} {r['max_abs_err']:>9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep BEAST num_basis/degree on cached batches.")
    parser.add_argument("--num-basis-grid", type=str, default="4,5,6,8")
    parser.add_argument("--degree-grid", type=str, default="2,3")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fit-beast-max-samples", type=int, default=1000, help="Train batches cached for fitting.")
    parser.add_argument("--max-eval-samples", type=int, default=300, help="Eval batches cached per eval dataset.")
    parser.add_argument("--eval-datasets", type=str, default="", help="Comma-separated subset of eval dataset names (default: all).")
    parser.add_argument("--num-dof", type=int, default=None, help="Leading action dims to tokenize (default: all).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="sweep_results")
    parser.add_argument("--bpe-config", type=str, default="", help="'NB,DEG' pair to additionally train BPE on.")
    parser.add_argument("--bpe-vocab-size", type=int, default=2048)
    parser.add_argument("--fit-bpe-max-samples", type=int, default=25_000, help="Sequences used to train BPE (bounded by cached batches).")
    args = parser.parse_args()

    from train.data import prepare_dataloaders  # heavy import: dataset stack

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    example_actions, dataloader_train, dataloader_evals = prepare_dataloaders(args.batch_size, num_workers=args.num_workers)
    seq_len, actions_dof = example_actions.shape
    num_dof = actions_dof if args.num_dof is None else args.num_dof
    if not 1 <= num_dof <= actions_dof:
        raise ValueError(f"--num-dof must be in [1, {actions_dof}], got {num_dof}")
    wanted = {name.strip() for name in args.eval_datasets.split(",") if name.strip()}
    eval_loaders = {name: dl for name, dl in dataloader_evals.items() if not wanted or name in wanted}
    print(f"chunk length={seq_len} action dims={actions_dof} tokenized dims={num_dof} eval datasets={list(eval_loaders)}", flush=True)

    t0 = time.time()
    fit_batches = _cache_batches(dataloader_train, args.fit_beast_max_samples)
    eval_batches = {name: _cache_batches(dl, args.max_eval_samples) for name, dl in eval_loaders.items()}
    print(
        f"cached {len(fit_batches)} fit batches and "
        + ", ".join(f"{name}: {len(b)}" for name, b in eval_batches.items())
        + f" eval batches in {time.time() - t0:.1f}s",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    summary_json = out_dir / "summary.json"
    if summary_json.exists():
        rows = json.load(open(summary_json))
        print(f"resuming: {len(rows)} rows already in {summary_json}")
    done = {(r["num_basis"], r["degree"], r["dataset"], r["is_bpe"]) for r in rows}

    tokenizers: Dict[str, BEASTBsplineTokenizer] = {}
    for nb, deg in itertools.product(_parse_grid(args.num_basis_grid), _parse_grid(args.degree_grid)):
        config = f"nb{nb}_deg{deg}"
        if nb < deg + 1:
            print(f"skip {config}: num_basis < degree + 1")
            continue
        if nb > seq_len:
            print(f"skip {config}: num_basis > chunk length {seq_len}")
            continue
        t0 = time.time()
        tokenizer = BEASTBsplineTokenizer(
            num_dof=num_dof, num_basis=nb, seq_len=seq_len, vocab_size=args.vocab_size,
            degree_p=deg, init_pos=False, device=args.device,
        )
        tokenizer.fit_parameters(fit_batches, max_samples=len(fit_batches), verbose=False)
        tokenizer.save_pretrained(out_dir / config / "beast_tokenizer_checkpoint")
        tokenizers[config] = tokenizer
        for name, batches in eval_batches.items():
            if (nb, deg, name, False) in done:
                continue
            stats = evaluate_tokenizer(tokenizer, batches, name, save_path=out_dir / config / "eval_results",
                                       max_eval_samples=len(batches), plot=False)
            row = _row(config, nb, deg, args.vocab_size, num_dof, name, stats)
            rows.append(row)
            _write_summary(rows, out_dir)
            print(f"{config} {name}: tokens={row['mean_tokens']:.1f} l1={row['mean_l1']:.5f} l2={row['mean_l2']:.6f} max_abs={row['max_abs_err']:.4f}", flush=True)
        print(f"{config} done in {time.time() - t0:.1f}s", flush=True)

    if args.bpe_config:
        nb, deg = _parse_grid(args.bpe_config)
        config = f"nb{nb}_deg{deg}"
        base = tokenizers.get(config)
        if base is None:
            base = BEASTBsplineTokenizer.from_pretrained(out_dir / config / "beast_tokenizer_checkpoint", device=args.device)
        bpe = BEASTBsplineBPETokenizer.from_beast(base, bpe_vocab_size=args.bpe_vocab_size)
        t0 = time.time()
        bpe.fit_from_trajectories(fit_batches, max_sequences=args.fit_bpe_max_samples)
        bpe.save_pretrained(out_dir / config / "beast_bpe_tokenizer_checkpoint")
        print(f"{config} BPE trained in {time.time() - t0:.1f}s", flush=True)
        for name, batches in eval_batches.items():
            if (nb, deg, name, True) in done:
                continue
            stats = evaluate_tokenizer(bpe, batches, name, save_path=out_dir / config / "eval_results_bpe",
                                       max_eval_samples=len(batches), plot=False)
            row = _row(config, nb, deg, args.vocab_size, num_dof, name, stats)
            rows.append(row)
            _write_summary(rows, out_dir)
            print(f"{config}+BPE {name}: tokens={row['mean_tokens']:.1f} l1={row['mean_l1']:.5f}", flush=True)

    print()
    _print_table(rows)
    print(f"\nsummary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
