"""Synthetic smoke test for the BEAST tokenizer stack (no dataset dependencies).

Run from the repository root on a machine with torch installed:

    PYTHONPATH=.:MP_lite_PyTorch python train/smoke_synthetic.py

The script builds sine-like action chunks of shape [B, 10, 32] (dims 26..31 are
zero padding, mimicking the unified 32-dim action space), then checks the base
B-spline tokenizer, checkpoint round-trips, the BPE extension, the evaluation
helper and the constructor guards. It exits non-zero on the first failure and
prints ``SMOKE OK`` on success.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "MP_lite_PyTorch"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import torch  # noqa: E402

from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer  # noqa: E402
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer  # noqa: E402
from train.eval import evaluate_tokenizer  # noqa: E402
from train.sweep_beast import load_batch_cache, save_batch_cache  # noqa: E402

SEQ_LEN = 10
NUM_DOF = 32
REAL_DOF = 26
NUM_BASIS = 5
DEGREE = 3
VOCAB = 256
BATCH = 64
NUM_BATCHES = 6


def make_batches(seed: int = 0, num_batches: int = NUM_BATCHES):
    gen = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 2.0 * math.pi, SEQ_LEN)
    batches = []
    for _ in range(num_batches):
        amp = torch.rand(BATCH, 1, REAL_DOF, generator=gen) * 0.9 + 0.1
        freq = torch.rand(BATCH, 1, REAL_DOF, generator=gen) * 0.7 + 0.3
        phase = torch.rand(BATCH, 1, REAL_DOF, generator=gen) * 2.0 * math.pi
        offset = (torch.rand(BATCH, 1, REAL_DOF, generator=gen) - 0.5) * 0.4
        real = amp * torch.sin(freq * t[None, :, None] + phase) + offset
        actions = torch.zeros(BATCH, SEQ_LEN, NUM_DOF)
        actions[..., :REAL_DOF] = real
        batches.append({"actions": actions})
    return batches


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def build_base(device: str = "cpu") -> BEASTBsplineTokenizer:
    return BEASTBsplineTokenizer(
        num_dof=NUM_DOF,
        num_basis=NUM_BASIS,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        degree_p=DEGREE,
        init_pos=False,
        device=device,
    )


def test_base_tokenizer(batches, tmp: Path):
    tok = build_base()
    tok.fit_parameters(batches, max_samples=len(batches), verbose=False)
    actions = batches[0]["actions"]

    tokens, _ = tok.encode(actions)
    check(tuple(tokens.shape) == (BATCH, NUM_BASIS * NUM_DOF), f"encode shape {tuple(tokens.shape)}")
    check(int(tokens.min()) >= 0 and int(tokens.max()) <= VOCAB - 1, "tokens outside [0, vocab)")

    # Zero-padded dims must map to a constant token (layout is 'b (t d)').
    for d in range(REAL_DOF, NUM_DOF):
        col = tokens[:, d::NUM_DOF]
        check(bool((col == col[0, 0]).all()), f"padded dim {d} is not constant")

    recon = tok.reconstruct_traj(tokens)
    check(tuple(recon.shape) == tuple(actions.shape), f"reconstruct shape {tuple(recon.shape)}")
    diff = (actions.to(recon.device) - recon).abs()
    expected_l1 = float(diff.mean())
    expected_l2 = float((diff ** 2).mean())
    expected_max = float(diff.amax())
    check(expected_l1 < 0.15, f"reconstruction too poor: l1={expected_l1:.4f}")

    metrics = tok.compute_reconstruction_metrics(actions)
    for key in ("l2", "l1", "max_abs", "per_dim_max_abs", "tokens"):
        check(key in metrics, f"metrics missing {key}")
    check(abs(float(metrics["l1"]) - expected_l1) < 1e-6, "l1 must be mean absolute error")
    check(abs(float(metrics["l2"]) - expected_l2) < 1e-6, "l2 must be mean squared error")
    check(abs(float(metrics["max_abs"]) - expected_max) < 1e-6, "max_abs mismatch")
    check(tuple(metrics["per_dim_max_abs"].shape) == (NUM_DOF,), "per_dim_max_abs shape")
    check(torch.equal(metrics["tokens"], tokens), "metrics tokens differ from encode")

    l2, l1 = tok.compute_reconstruction_error(actions)
    check(abs(float(l1) - expected_l1) < 1e-6, "compute_reconstruction_error l1 is not abs-mean")
    l2b, l1b, toks = tok.compute_reconstruction_error(actions, return_tokens=True)
    check(torch.equal(toks, tokens), "return_tokens must return encode tokens")

    # Checkpoint round-trip for the base class.
    ckpt = tmp / "base_ckpt"
    tok.save_pretrained(ckpt)
    loaded = BEASTBsplineTokenizer.from_pretrained(ckpt, device="cpu")
    tokens2, _ = loaded.encode(actions)
    check(torch.equal(tokens2, tokens), "base from_pretrained round-trip changed tokens")
    return tok


def test_bpe_tokenizer(base: BEASTBsplineTokenizer, batches, tmp: Path):
    bpe = BEASTBsplineBPETokenizer.from_beast(base, bpe_vocab_size=512)
    bpe.fit_from_trajectories(batches, max_sequences=BATCH * len(batches), show_progress=False)
    actions = batches[1]["actions"]

    mp_tokens, _ = bpe.encode_to_mp_tokens(actions)
    bpe_tokens, _ = bpe.encode(actions)
    check(isinstance(bpe_tokens, list) and len(bpe_tokens) == BATCH, "bpe encode must return a list per sample")
    back = bpe.bpe_to_mp_tokens(bpe_tokens)
    check(torch.equal(back.cpu(), mp_tokens.cpu()), "bpe round-trip changed MP tokens")
    mean_len = sum(len(row) for row in bpe_tokens) / len(bpe_tokens)
    check(mean_len < NUM_BASIS * NUM_DOF, f"BPE did not compress: mean len {mean_len}")

    recon = bpe.reconstruct_traj(bpe_tokens)
    check(tuple(recon.shape) == tuple(actions.shape), "bpe reconstruct shape")
    l2, l1, toks = bpe.compute_reconstruction_error(actions, return_tokens=True)
    check(isinstance(toks, list), "bpe compute_reconstruction_error must return BPE token lists")
    check(float(l1) > 0.0, "bpe l1 must be positive mean absolute error")

    ckpt = tmp / "bpe_ckpt"
    bpe.save_pretrained(ckpt)
    loaded = BEASTBsplineBPETokenizer.from_pretrained(ckpt, device="cpu")
    bpe_tokens2, _ = loaded.encode(actions)
    check(bpe_tokens2 == bpe_tokens, "bpe from_pretrained round-trip changed tokens")
    return bpe


def test_evaluate(base, bpe, batches, tmp: Path):
    stats = evaluate_tokenizer(base, batches, "synthetic_base", save_path=tmp / "eval", max_eval_samples=len(batches), plot=False)
    for key in ("mean_l2", "mean_l1", "mean_max_abs", "tokens_pre_bpe", "mean_tokens", "is_bpe"):
        check(key in stats, f"stats missing {key}")
    check(stats["tokens_pre_bpe"] == NUM_BASIS * NUM_DOF, f"tokens_pre_bpe {stats['tokens_pre_bpe']}")
    check(stats["mean_tokens"] == NUM_BASIS * NUM_DOF, "base tokenizer token length must be fixed")
    check(stats["is_bpe"] is False, "base tokenizer is not BPE")
    check((tmp / "eval" / "synthetic_base" / "stats.txt").exists(), "stats.txt not written")
    check((tmp / "eval" / "synthetic_base" / "errors.json").exists(), "errors.json not written")

    stats_bpe = evaluate_tokenizer(bpe, batches, "synthetic_bpe", save_path=tmp / "eval", max_eval_samples=len(batches), plot=False)
    check(stats_bpe["is_bpe"] is True, "bpe tokenizer flag")
    check(stats_bpe["tokens_pre_bpe"] == NUM_BASIS * NUM_DOF, "bpe tokens_pre_bpe")
    check(stats_bpe["mean_tokens"] < NUM_BASIS * NUM_DOF, "bpe mean_tokens must be below pre-BPE length")


def test_batch_cache(batches, tmp: Path):
    eval_batches = {"synthetic_a": batches[:2], "synthetic_b": batches[2:3]}
    meta = {"seq_len": SEQ_LEN, "actions_dof": NUM_DOF, "batch_size": BATCH}
    path = tmp / "batches.pt"
    save_batch_cache(path, batches, eval_batches, meta)
    check(path.exists(), "batch cache file not written")
    fit2, eval2, meta2 = load_batch_cache(path)
    check(len(fit2) == len(batches), "fit batch count changed after reload")
    check(all(torch.equal(a["actions"], b["actions"]) for a, b in zip(fit2, batches)), "fit batches changed after reload")
    check(set(eval2) == set(eval_batches), "eval dataset names changed after reload")
    check(torch.equal(eval2["synthetic_b"][0]["actions"], batches[2]["actions"]), "eval batches changed after reload")
    check(meta2["seq_len"] == SEQ_LEN and meta2["actions_dof"] == NUM_DOF, "meta changed after reload")


def test_guards():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BEASTBsplineTokenizer(num_dof=NUM_DOF, num_basis=50, seq_len=SEQ_LEN, vocab_size=1000, degree_p=0, init_pos=False, device="cpu")
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    check(any("num_basis" in m for m in msgs), "num_basis > seq_len must warn")

    try:
        BEASTBsplineTokenizer(num_dof=NUM_DOF, num_basis=3, seq_len=SEQ_LEN, vocab_size=VOCAB, degree_p=3, init_pos=False, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("num_basis < degree_p + 1 must raise ValueError")


def main() -> int:
    torch.manual_seed(0)
    batches = make_batches()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base = test_base_tokenizer(batches, tmp)
        print("base tokenizer: OK")
        bpe = test_bpe_tokenizer(base, batches, tmp)
        print("bpe tokenizer: OK")
        test_evaluate(base, bpe, batches, tmp)
        print("evaluate_tokenizer: OK")
        test_batch_cache(batches, tmp)
        print("batch cache: OK")
    test_guards()
    print("constructor guards: OK")
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
