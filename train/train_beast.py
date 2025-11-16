"""Train BEAST tokenizers (with optional BPE stage)."""
import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from torch.utils.data import DataLoader

from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer
from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer
from train.data import prepare_dataloaders


def _limit_batches(loader: DataLoader, max_batches: Optional[int]) -> Iterator[Any]:
    if max_batches is None or max_batches <= 0:
        yield from loader
        return

    for idx, batch in enumerate(loader):
        yield batch
        if (idx + 1) >= max_batches:
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the base BEAST tokenizer and optionally the BEAST+BPE extension."
        )
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataloaders.")
    parser.add_argument("--num-basis", type=int, default=32, help="Number of spline basis functions.")
    parser.add_argument("--vocab-size", type=int, default=800, help="Discrete vocab size.")
    parser.add_argument("--degree", type=int, default=1, help="Spline degree p.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for fitting (cpu or cuda).")
    parser.add_argument("--gripper-indices", type=int, nargs="*", default=(), help="Optional gripper DoF indices.")
    parser.add_argument("--fit-beast-max-samples", type=int, default=BEAST_TRAIN_MAX_SAMPLES, help="Number of sequences for BEAST parameter fitting.")
    parser.add_argument("--fit-bpe-max-samples",type=int,default=BPE_TRAIN_MAX_SAMPLES,help="Number of dataloader batches used for BPE fitting.")
    parser.add_argument("--bpe-vocab-size", type=int, default=2048, help="Vocabulary size for the BPE tokenizer.")
    parser.add_argument("--beast-checkpoint-dir", type=str, default="beast_tokenizer_checkpoint", help="Directory to store the base tokenizer.")
    parser.add_argument("--bpe-checkpoint-dir", type=str, default="beast_bpe_tokenizer_checkpoint", help="Directory to store the BPE tokenizer.")
    
    train_bpe_group = parser.add_mutually_exclusive_group()
    train_bpe_group.add_argument( "--train-bpe", dest="train_bpe", action="store_true", help="Train the BPE tokenizer (enabled by default).")
    train_bpe_group.add_argument( "--no-train-bpe", dest="train_bpe", action="store_false", help="Skip the BPE tokenizer fitting stage.")
    parser.set_defaults(train_bpe=True)
    
    args = parser.parse_args()

    example_actions, dataloader_train, dataloader_evals = prepare_dataloaders(args.batch_size)
    actions_len, actions_dof = example_actions.shape

    # ===============================================================================
    #                           - BEAST tokenizer fitting -                          
    # ===============================================================================
    tokenizer = BEASTBsplineTokenizer(
        num_basis=args.num_basis,
        vocab_size=args.vocab_size,
        degree_p=args.degree,
        num_dof=actions_dof,
        seq_len=actions_len,
        gripper_indices=list(args.gripper_indices),
        gripper_zero_order=False,
        init_pos=False,
        device=args.device,
    )
    
    tokenizer.fit_parameters(dataloader_train, max_samples=args.fit_beast_max_samples)
    Path(args.beast_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.beast_checkpoint_dir)
    print(f"Saved BEAST tokenizer to {args.beast_checkpoint_dir}")
    
    
    # ===============================================================================
    #                           - BPE tokenizer fitting -                          
    # ===============================================================================

    if not args.train_bpe:
        print("Skipping BPE training (use --train-bpe to enable).")
    
    else:
        bpe_tokenizer = BEASTBsplineBPETokenizer.from_beast(
            tokenizer, bpe_vocab_size=args.bpe_vocab_size
        )
        limited_batches: Iterable[Any] = _limit_batches(dataloader_train, args.max_samples)
        bpe_tokenizer.fit_from_trajectories(limited_batches)
        Path(args.bpe_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        bpe_tokenizer.save_pretrained(args.bpe_checkpoint_dir)
        print(f"Saved BEAST+BPE tokenizer to {args.bpe_checkpoint_dir}")
    
    # ===============================================================================
    #                           - Evaluating tokenizer -                          
    # ===============================================================================
    
    for dts_name, dataloader_eval in dataloader_evals.items():
        print(f"Evaluating {dts_name} tokenizer")
        bpe_tokenizer.evaluate(dataloader_eval)


if __name__ == "__main__":
    main()
