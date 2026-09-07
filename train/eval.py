"""Reconstruction-error evaluation for BEAST tokenizers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:  # seaborn is optional: fall back to plain matplotlib histograms
    import seaborn as sns
except ImportError:  # pragma: no cover - depends on the environment
    sns = None

from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer


def _summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "max": nan, "min": nan}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def _hist(ax, values: List[float], log_scale: bool, title: str, xlabel: str) -> None:
    arr = np.asarray(values, dtype=float)
    if log_scale:
        arr = arr[arr > 0]
    if arr.size == 0:
        ax.set_title(f"{title} (no data)")
        return
    if sns is not None:
        sns.histplot(arr, bins=100, alpha=0.5, color="b", kde=True, ax=ax, log_scale=(log_scale, False))
    else:
        if log_scale and arr.min() < arr.max():
            bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 100)
            ax.set_xscale("log")
        else:
            bins = 100
        ax.hist(arr, bins=bins, alpha=0.5, color="b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def _plot_pair(values: List[float], name: str, save_file: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _hist(ax1, values, False, f"{name} Distribution (Linear Scale)", name)
    _hist(ax2, values, True, f"{name} Distribution (Log Scale)", f"{name} (log scale)")
    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    plt.close(fig)


def evaluate_tokenizer(
    tokenizer: BEASTBsplineTokenizer,
    dataloader: Iterable[dict],
    dataset_name: str,
    save_path: str | Path = "eval_results",
    max_eval_samples: int = 12_500,
    plot: bool = True,
) -> Dict[str, Any]:
    """Compute reconstruction and token-length statistics of ``tokenizer`` on ``dataloader``.

    ``max_eval_samples`` counts dataloader batches. Writes ``errors.json`` and ``stats.txt``
    (plus histograms when ``plot``) under ``save_path/dataset_name`` and returns the stats.
    """
    save_dir = Path(save_path) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    errors_l2: List[float] = []
    errors_l1: List[float] = []
    errors_max_abs: List[float] = []
    token_lengths: List[int] = []
    for batch in tqdm(dataloader, total=max_eval_samples, desc=f"Reconstruction errors [{dataset_name}]"):
        if len(errors_l2) >= max_eval_samples:
            break
        metrics = tokenizer.compute_reconstruction_metrics(batch["actions"])
        errors_l2.append(float(metrics["l2"]))
        errors_l1.append(float(metrics["l1"]))
        errors_max_abs.append(float(metrics["max_abs"]))
        for row in metrics["tokens"]:
            token_lengths.append(int(len(row)))

    l2, l1, mx, tl = _summary(errors_l2), _summary(errors_l1), _summary(errors_max_abs), _summary(token_lengths)
    stats: Dict[str, Any] = {
        "dataset": dataset_name,
        "num_batches": len(errors_l2),
        "num_chunks": len(token_lengths),
        "is_bpe": isinstance(tokenizer, BEASTBsplineBPETokenizer),
        "tokens_pre_bpe": int(tokenizer.num_basis * tokenizer.num_dof),
        "mean_tokens": tl["mean"],
        "std_tokens": tl["std"],
        "max_tokens": tl["max"],
        "min_tokens": tl["min"],
        "mean_l2": l2["mean"],
        "std_l2": l2["std"],
        "max_l2": l2["max"],
        "min_l2": l2["min"],
        "mean_l1": l1["mean"],
        "std_l1": l1["std"],
        "max_l1": l1["max"],
        "min_l1": l1["min"],
        "mean_max_abs": mx["mean"],
        "max_max_abs": mx["max"],
    }

    with open(save_dir / "errors.json", "w") as f:
        json.dump(
            {
                "errors_l2": errors_l2,
                "errors_l1": errors_l1,
                "errors_max_abs": errors_max_abs,
                "mean_tokens_length": token_lengths,
            },
            f,
        )

    with open(save_dir / "stats.txt", "w") as f:
        print("Tokenizer is BPE:", stats["is_bpe"], file=f)
        print("Tokens per chunk before BPE:", stats["tokens_pre_bpe"], file=f)
        print("Mean tokens length:", stats["mean_tokens"], file=f)
        print("Std tokens length:", stats["std_tokens"], file=f)
        print("Max tokens length:", stats["max_tokens"], file=f)
        print("Min tokens length:", stats["min_tokens"], file=f)
        print("", file=f)
        for key in ("l2", "l1"):
            print(f"Mean reconstruction error {key}:", stats[f"mean_{key}"], file=f)
            print(f"Std reconstruction error {key}:", stats[f"std_{key}"], file=f)
            print(f"Max reconstruction error {key}:", stats[f"max_{key}"], file=f)
            print(f"Min reconstruction error {key}:", stats[f"min_{key}"], file=f)
            print("", file=f)
        print("Mean of per-batch max abs error:", stats["mean_max_abs"], file=f)
        print("Max abs error:", stats["max_max_abs"], file=f)

    if plot:
        _plot_pair(errors_l2, "L2 Error", save_dir / "histogram_l2.png")
        _plot_pair(errors_l1, "L1 Error", save_dir / "histogram_l1.png")
        _plot_pair([float(t) for t in token_lengths], "Tokens Length", save_dir / "histogram_mean_tokens_length.png")

    return stats


def evaluate_from_path(
    dataloader: Iterable[dict],
    dataset_name: str,
    tokenizer_path: str,
    is_bpe_tokenizer: bool = True,
    save_path: str | Path = "eval_results",
    max_eval_samples: int = 12_500,
    plot: bool = True,
) -> Dict[str, Any]:
    """Load a saved tokenizer from ``tokenizer_path`` and evaluate it (see ``evaluate_tokenizer``)."""
    if is_bpe_tokenizer:
        tokenizer = BEASTBsplineBPETokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = BEASTBsplineTokenizer.from_pretrained(tokenizer_path)
    return evaluate_tokenizer(
        tokenizer,
        dataloader,
        dataset_name,
        save_path=save_path,
        max_eval_samples=max_eval_samples,
        plot=plot,
    )
