from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer

from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer


ArrayLike = Union[Sequence[int], np.ndarray, torch.Tensor]


def _flatten_to_numpy(sequence: ArrayLike) -> np.ndarray:
    if isinstance(sequence, torch.Tensor):
        array = sequence.detach().cpu().numpy()
    else:
        array = np.asarray(sequence)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.astype(np.int64)


@dataclass
class FIGBPEState:
    tokenizer: ByteLevelBPETokenizer
    min_token: int
    max_token: int


class FIGBPE:
    """Trainer for Byte Pair Encoding over discretised BEAST tokens."""

    def __init__(
        self,
        vocab_size: int = 1024,
        *,
        min_frequency: int = 2,
        special_tokens: Optional[Sequence[str]] = None,
        show_progress: bool = True,
        max_token_length: int = 10000,
    ) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = list(special_tokens or [])
        self.show_progress = show_progress
        self.max_token_length = max_token_length

        self.tokenizer: Optional[ByteLevelBPETokenizer] = None
        self.min_token: Optional[int] = None
        self.max_token: Optional[int] = None

    def _fit_from_strings(
        self, strings: List[str], alphabet: Sequence[str]
    ) -> ByteLevelBPETokenizer:
        bpe = ByteLevelBPETokenizer()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            show_progress=self.show_progress,
            special_tokens=self.special_tokens,
            initial_alphabet=list(alphabet),
            max_token_length=self.max_token_length,
        )
        bpe._tokenizer.train_from_iterator(strings, trainer=trainer)
        return bpe

    def fit_from_sequences(self, sequences: Iterable[ArrayLike]) -> FIGBPEState:
        processed: List[np.ndarray] = []
        for seq in sequences:
            arr = _flatten_to_numpy(seq)
            if arr.size == 0:
                continue
            processed.append(arr)
        if not processed:
            raise ValueError("No non-empty sequences provided for BPE training.")

        min_token = int(min(int(arr.min()) for arr in processed))
        max_token = int(max(int(arr.max()) for arr in processed))

        normalized_strings = [
            "".join(map(chr, (arr - min_token).astype(int))) for arr in processed
        ]
        alphabet = [chr(i) for i in range(max_token - min_token + 1)]

        tokenizer = self._fit_from_strings(normalized_strings, alphabet)
        self.tokenizer = tokenizer
        self.min_token = min_token
        self.max_token = max_token
        return FIGBPEState(tokenizer=tokenizer, min_token=min_token, max_token=max_token)

    def fit_from_trajectories(
        self,
        tokenizer: BEASTBsplineTokenizer,
        trajectories: Iterable[Union[ArrayLike, dict]],
        *,
        update_bounds: bool = False,
        batch_key: str = "actions",
        max_sequences: Optional[int] = None,
    ) -> FIGBPEState:
        sequences: List[np.ndarray] = []
        collected = 0
        encode_fn = getattr(tokenizer, "encode_to_mp_tokens", None)
        if encode_fn is None:
            encode_fn = tokenizer.encode
        for batch in trajectories:
            if isinstance(batch, dict):
                if batch_key not in batch:
                    raise KeyError(
                        f"Batch dictionary is missing required key '{batch_key}'."
                    )
                data = batch[batch_key]
            else:
                data = batch

            if not torch.is_tensor(data):
                data = torch.as_tensor(data)
            data = data.to(tokenizer.device)

            tokens, _ = encode_fn(data, update_bounds=update_bounds)
            tokens_np = tokens.detach().cpu().numpy()
            for row in tokens_np:
                sequences.append(row.astype(np.int64))
                collected += 1
                if max_sequences is not None and collected >= max_sequences:
                    return self.fit_from_sequences(sequences)
        return self.fit_from_sequences(sequences)

    def get_state(self) -> FIGBPEState:
        if self.tokenizer is None or self.min_token is None or self.max_token is None:
            raise RuntimeError("BPE tokenizer has not been trained yet.")
        return FIGBPEState(
            tokenizer=self.tokenizer,
            min_token=self.min_token,
            max_token=self.max_token,
        )
