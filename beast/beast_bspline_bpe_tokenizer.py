from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer

from beast.beast_bspline_tokenizer import CONFIG_FILENAME, BEASTBsplineTokenizer


TokenLike = Union[Sequence[int], torch.Tensor, np.ndarray]


if TYPE_CHECKING:
    from beast.beast_bpe_trainer import FIGBPEState


class BEASTBsplineBPETokenizer(BEASTBsplineTokenizer):
    """B-Spline tokenizer augmented with a learned Byte-Pair encoder."""

    bpe_subdir = "bpe_tokenizer"

    def __init__(
        self,
        *args,
        bpe_vocab_size: int = 1024,
        bpe_min_token: int = 0,
        base_tokenizer: Optional[BEASTBsplineTokenizer] = None,
        **kwargs,
    ) -> None:
        kwargs = kwargs.copy()
        kwargs.pop("use_bpe", None)
        kwargs.pop("tokenizer_type", None)

        self.bpe_vocab_size = bpe_vocab_size
        self.bpe_tokenizer: Optional[ByteLevelBPETokenizer] = None
        self.bpe_min_token: int = int(bpe_min_token)
        self.bpe_max_token: Optional[int] = None

        if base_tokenizer is not None:
            if args:
                raise TypeError(
                    "Positional arguments are not supported when base_tokenizer is provided."
                )
            if not isinstance(base_tokenizer, BEASTBsplineTokenizer):
                raise TypeError("base_tokenizer must be a BEASTBsplineTokenizer instance.")
            base_state = base_tokenizer.state_dict()
            base_config = base_state.get("config", {}).copy()
            base_config.pop("tokenizer_type", None)
            base_config["use_bpe"] = True
            device_override = kwargs.pop("device", None)
            if kwargs:
                unexpected = ", ".join(sorted(kwargs.keys()))
                raise TypeError(
                    "Unexpected keyword arguments when base_tokenizer is provided: "
                    f"{unexpected}."
                )
            if device_override is not None:
                base_config["device"] = device_override
            super().__init__(**base_config)
        else:
            super().__init__(*args, use_bpe=True, **kwargs)

        self._config["bpe_vocab_size"] = bpe_vocab_size
        self._config["tokenizer_type"] = "beast_bspline_bpe"
        self._config["bpe_min_token"] = self.bpe_min_token

        if base_tokenizer is not None:
            self.load_state_dict(base_state)
            
    def to(self, device: str | torch.device) -> BEASTBsplineBPETokenizer:
        super().to(device)
        self.device = device
        return self


    # ===============================================
    #                  - utilities -          
    # ===============================================
    
    def _require_bpe(self) -> ByteLevelBPETokenizer:
        if self.bpe_tokenizer is None:
            raise RuntimeError(
                "BPE tokenizer has not been trained. Call fit_from_trajectories() "
                "or set_bpe_tokenizer() with a trained tokenizer."
            )
        return self.bpe_tokenizer

    @property
    def sequence_length(self) -> int:
        return self.num_basis * self.num_dof

    def set_bpe_tokenizer(
        self,
        tokenizer: ByteLevelBPETokenizer,
        *,
        min_token: int = 0,
        max_token: Optional[int] = None,
    ) -> None:
        if not isinstance(tokenizer, ByteLevelBPETokenizer):
            raise TypeError("Expected a ByteLevelBPETokenizer instance.")
        self.bpe_tokenizer = tokenizer
        self.bpe_min_token = int(min_token)
        self.bpe_max_token = None if max_token is None else int(max_token)
        self._config["bpe_min_token"] = self.bpe_min_token

    def fit_from_trajectories(
        self,
        trajectories: Iterable[Union[TokenLike, dict]],
        *,
        update_bounds: bool = False,
        batch_key: str = "actions",
        max_sequences: Optional[int] = None,
        min_frequency: int = 2,
        special_tokens: Optional[Sequence[str]] = None,
        show_progress: bool = True,
        max_token_length: int = 10000,
    ) -> "FIGBPEState":
        """Train the internal BPE model using BEAST discretised tokens."""

        from beast.beast_bpe_trainer import FIGBPE

        fig_bpe = FIGBPE(
            vocab_size=self.bpe_vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=show_progress,
            max_token_length=max_token_length,
        )
        state = fig_bpe.fit_from_trajectories(
            self,
            trajectories,
            update_bounds=update_bounds,
            batch_key=batch_key,
            max_sequences=max_sequences,
        )
        self.set_bpe_tokenizer(
            state.tokenizer,
            min_token=state.min_token,
            max_token=state.max_token,
        )
        return state


    # ===============================================
    #             - encoding / decoding -          
    # ===============================================

    def _as_sequence_list(self, values: TokenLike) -> List[np.ndarray]:
        sequences: List[np.ndarray]
        if isinstance(values, torch.Tensor):
            if values.ndim == 1:
                sequences = [values.detach().cpu().numpy()]
            elif values.ndim == 2:
                sequences = [row.detach().cpu().numpy() for row in values]
            else:
                raise ValueError("Expected tensor with 1 or 2 dimensions for token sequences.")
        elif isinstance(values, np.ndarray):
            if values.ndim == 1:
                sequences = [values]
            elif values.ndim == 2:
                sequences = [row for row in values]
            else:
                raise ValueError("Expected numpy array with 1 or 2 dimensions for token sequences.")
        elif isinstance(values, Sequence) and values and isinstance(values[0], numbers.Integral):
            sequences = [np.asarray(values)]
        else:
            sequences = [np.asarray(row) for row in values]  # type: ignore[arg-type]
        return sequences

    def _discrete_to_bpe(self, discrete_tokens: TokenLike) -> List[List[int]]:
        tokenizer = self._require_bpe()
        sequences = self._as_sequence_list(discrete_tokens)
        result: List[List[int]] = []
        for seq in sequences:
            flattened = np.asarray(seq).reshape(-1).astype(int)
            shifted = flattened - self.bpe_min_token
            if (shifted < 0).any():
                raise ValueError(
                    "Discrete tokens contain values smaller than the configured BPE minimum token."
                )
            if self.bpe_max_token is not None:
                max_allowed = self.bpe_max_token - self.bpe_min_token
                if (shifted > max_allowed).any():
                    raise ValueError(
                        "Discrete tokens contain values greater than the configured BPE maximum token. "
                        "Either retrain the BPE tokenizer with a wider range or disable BPE for this run."
                    )
            text = "".join(map(chr, shifted))
            # Disable automatic BOS/EOS insertion because their empty decoded
            # representations would make it impossible to reconstruct the exact
            # sequence length when converting back to discrete tokens.
            result.append(tokenizer.encode(text, add_special_tokens=False).ids)
        return result

    def _bpe_to_discrete(self, tokens: Iterable[TokenLike]) -> torch.Tensor:
        tokenizer = self._require_bpe()
        if isinstance(tokens, torch.Tensor):
            token_sequences: Iterable[TokenLike]
            if tokens.ndim == 1:
                token_sequences = [tokens]
            elif tokens.ndim == 2:
                token_sequences = [row for row in tokens]
            else:
                raise ValueError("Expected tensor with 1 or 2 dimensions for BPE tokens.")
        elif isinstance(tokens, np.ndarray):
            if tokens.ndim == 1:
                token_sequences = [tokens]
            elif tokens.ndim == 2:
                token_sequences = [row for row in tokens]
            else:
                raise ValueError("Expected numpy array with 1 or 2 dimensions for BPE tokens.")
        elif isinstance(tokens, Sequence) and tokens and isinstance(tokens[0], numbers.Integral):
            token_sequences = [tokens]
        else:
            token_sequences = tokens

        sequences: List[np.ndarray] = []
        unk_id = tokenizer.token_to_id("<unk>")
        for token in token_sequences:
            if isinstance(token, torch.Tensor):
                token_list = [int(t) for t in token.detach().cpu().tolist()]
            elif isinstance(token, np.ndarray):
                token_list = [int(t) for t in token.tolist()]
            else:
                token_list = [int(t) for t in token]
            if unk_id is not None and unk_id in token_list:
                raise ValueError(
                    "BPE sequence contains <unk> tokens. This usually means that the discrete "
                    "BEAST tokens went out of the range seen during BPE training. Consider "
                    "retraining the BPE tokenizer with a wider token range or disable BPE."
                )
            # Skip special tokens when decoding to avoid empty placeholders
            # (e.g. <s>, </s>) shortening the reconstructed sequence.
            text = tokenizer.decode(token_list, skip_special_tokens=True)
            decoded = np.array(list(map(ord, text)), dtype=np.int64) + self.bpe_min_token
            if decoded.size != self.sequence_length:
                raise ValueError(
                    f"Decoded sequence has length {decoded.size}, expected {self.sequence_length}."
                )
            sequences.append(decoded)
        stacked = torch.tensor(np.stack(sequences), dtype=torch.long, device=self.device)
        return stacked


    # ===============================================
    #               - BEAST overriden -          
    # ===============================================
    
    def encode(
        self,
        trajs: torch.Tensor,
        update_bounds: bool = False,
        *,
        return_mp_tokens: bool = False,
    ) -> tuple:
        mp_tokens, params = super().encode(
            trajs,
            update_bounds=update_bounds,
            respect_llm_vocab_size=False,
        )
        bpe_tokens = self._discrete_to_bpe(mp_tokens)
        if return_mp_tokens:
            return bpe_tokens, params, mp_tokens
        return bpe_tokens, params

    def decode(
        self,
        tokens: Iterable[TokenLike],
        *,
        respect_llm_vocab_size: bool = False,
    ) -> torch.Tensor:
        discrete = self._bpe_to_discrete(tokens)
        return super().decode(
            discrete,
            respect_llm_vocab_size=respect_llm_vocab_size,
        )

    def encode_to_mp_tokens(
        self, trajs: torch.Tensor, update_bounds: bool = False
    ) -> tuple:
        """Expose the underlying MP-token encoding without BPE."""
        return super().encode(
            trajs,
            update_bounds=update_bounds,
            respect_llm_vocab_size=False,
        )

    def bpe_to_mp_tokens(self, tokens: Iterable[TokenLike]) -> torch.Tensor:
        """Convert BPE tokens back to discrete BEAST bins."""
        return self._bpe_to_discrete(tokens)

    def reconstruct_traj(
        self,
        tokens: Iterable[TokenLike],
        times: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # discrete = self._bpe_to_discrete(tokens)
        return super().reconstruct_traj(tokens, times=times, **kwargs)


    # ===============================================
    #                 - serialization -          
    # ===============================================
    
    def get_config(self):  # type: ignore[override]
        config = super().get_config()
        config["bpe_vocab_size"] = self.bpe_vocab_size
        config["use_bpe"] = True
        return config

    def state_dict(self):  # type: ignore[override]
        state = super().state_dict()
        state["bpe"] = {
            "min_token": self.bpe_min_token,
            "max_token": self.bpe_max_token,
            "vocab_size": self.bpe_vocab_size,
            "tokenizer_dir": self.bpe_subdir if self.bpe_tokenizer is not None else None,
        }
        return state

    def load_state_dict(self, state_dict):  # type: ignore[override]
        super().load_state_dict(state_dict)
        bpe_info = state_dict.get("bpe", {})
        self.bpe_min_token = int(bpe_info.get("min_token", self.bpe_min_token))
        max_token = bpe_info.get("max_token", self.bpe_max_token)
        self.bpe_max_token = None if max_token is None else int(max_token)
        self.bpe_vocab_size = int(bpe_info.get("vocab_size", self.bpe_vocab_size))
        self._config["bpe_min_token"] = self.bpe_min_token

    def save_pretrained(self, save_directory):  # type: ignore[override]
        save_directory = Path(save_directory)
        super().save_pretrained(save_directory)

        if self.bpe_tokenizer is not None:
            bpe_dir = save_directory / self.bpe_subdir
            bpe_dir.mkdir(parents=True, exist_ok=True)
            files = self.bpe_tokenizer.save_model(str(bpe_dir))
            self.bpe_tokenizer.save(str(bpe_dir / "tokenizer.json"))
            saved_files = ", ".join(Path(f).name for f in files)
            print(
                "  - BPE tokenizer files: "
                f"{saved_files} and tokenizer.json in {bpe_dir}"
            )

    @classmethod
    def from_pretrained(cls, pretrained_path, device=None):  # type: ignore[override]
        pretrained_path = Path(pretrained_path)
        config_path = pretrained_path / CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        config = state["config"]
        config = config.copy()
        tokenizer_type = config.get("tokenizer_type")
        if tokenizer_type not in {"beast_bspline_bpe", None}:
            raise ValueError(
                "Loaded configuration does not describe a BEAST B-Spline BPE tokenizer."
            )
        config["tokenizer_type"] = "beast_bspline_bpe"
        config["use_bpe"] = True
        if device is not None:
            config["device"] = device
        tokenizer = cls(**config)
        tokenizer.load_state_dict(state)
        bpe_info = state.get("bpe", {})
        bpe_dir_name = bpe_info.get("tokenizer_dir", cls.bpe_subdir)
        bpe_dir = pretrained_path / bpe_dir_name
        if bpe_dir.exists():
            vocab_path = bpe_dir / "vocab.json"
            merges_path = bpe_dir / "merges.txt"
            if vocab_path.exists() and merges_path.exists():
                tokenizer.bpe_tokenizer = ByteLevelBPETokenizer.from_file(
                    str(vocab_path),
                    str(merges_path),
                )
        tokenizer.bpe_min_token = int(bpe_info.get("min_token", tokenizer.bpe_min_token))
        max_token = bpe_info.get("max_token", tokenizer.bpe_max_token)
        tokenizer.bpe_max_token = None if max_token is None else int(max_token)
        tokenizer.bpe_vocab_size = int(bpe_info.get("vocab_size", tokenizer.bpe_vocab_size))
        tokenizer._config["bpe_min_token"] = tokenizer.bpe_min_token
        return tokenizer

    @classmethod
    def from_beast(
        cls,
        tokenizer: BEASTBsplineTokenizer,
        *,
        bpe_vocab_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "BEASTBsplineBPETokenizer":
        """Instantiate a BPE-enabled tokenizer from a fitted BEAST tokenizer."""

        if not isinstance(tokenizer, BEASTBsplineTokenizer):
            raise TypeError("tokenizer must be a BEASTBsplineTokenizer instance.")

        init_kwargs = {"base_tokenizer": tokenizer}
        if bpe_vocab_size is not None:
            init_kwargs["bpe_vocab_size"] = bpe_vocab_size
        if device is not None:
            init_kwargs["device"] = device
        return cls(**init_kwargs)

    @classmethod
    def from_bspline_tokenizer(
        cls,
        tokenizer: BEASTBsplineTokenizer,
        *,
        bpe_vocab_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "BEASTBsplineBPETokenizer":
        """Backward-compatible alias for :meth:`from_beast`."""

        return cls.from_beast(
            tokenizer,
            bpe_vocab_size=bpe_vocab_size,
            device=device,
        )
