"""Train BEAST tokenizers (with optional BPE stage)."""
import argparse
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional
import json

from torch.utils.data import DataLoader

from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer
from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer

from train.data import prepare_dataloaders
from train.eval import evaluate_from_path


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
    parser.add_argument("--num-basis", type=int, default=50, help="Number of spline basis functions.")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Discrete vocab size.")
    parser.add_argument("--degree", type=int, default=0, help="Spline degree p.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for fitting (cpu or cuda).")
    parser.add_argument("--fit-beast-max-samples", type=int, default=5_000, help="Number of sequences for BEAST parameter fitting.")
    parser.add_argument("--fit-bpe-max-samples",type=int,default=25_000, help="Number of dataloader batches used for BPE fitting.")
    parser.add_argument("--bpe-vocab-size", type=int, default=2048, help="Vocabulary size for the BPE tokenizer.")
    parser.add_argument("--beast-checkpoint-dir", type=str, default="beast_tokenizer_checkpoint", help="Directory to store the base tokenizer.")
    parser.add_argument("--bpe-checkpoint-dir", type=str, default="beast_bpe_tokenizer_checkpoint", help="Directory to store the BPE tokenizer.")
    parser.add_argument("--eval-results-dir", type=str, default="eval_results", help="Directory to store the evaluation results.")
    parser.add_argument("--max-eval-samples", type=int, default=12_500, help="Number of samples to evaluate on.")
    
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
        limited_batches: Iterable[Any] = _limit_batches(dataloader_train, args.fit_bpe_max_samples)
        bpe_tokenizer.fit_from_trajectories(limited_batches)
        Path(args.bpe_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        bpe_tokenizer.save_pretrained(args.bpe_checkpoint_dir)
        print(f"Saved BEAST+BPE tokenizer to {args.bpe_checkpoint_dir}")
    
    # ===============================================================================
    #                           - Evaluating tokenizer -                          
    # ===============================================================================
    
    total_stats = {}
    for dts_name, dataloader_eval in dataloader_evals.items():
        print(f"Evaluating {dts_name} tokenizer")
        
        if args.train_bpe:
            tokenizer_path = args.bpe_checkpoint_dir
        else:
            tokenizer_path = args.beast_checkpoint_dir
        
        stats = evaluate_from_path(
            dataloader_eval, 
            dts_name, 
            tokenizer_path,
            args.train_bpe,
            save_path=args.eval_results_dir,
            max_eval_samples=args.max_eval_samples,
        )
        
        total_stats[dts_name] = stats
        
    with open(Path(args.eval_results_dir) / 'total_stats.json', 'w') as f:
        json.dump(total_stats, f)

if __name__ == "__main__":
    main()
