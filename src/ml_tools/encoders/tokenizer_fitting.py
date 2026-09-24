"""Fit a SentencePiece model on a text corpus and serialize it to disk."""

import glob
from typing import Dict, List, Optional

import sentencepiece as spm
from ml_tools.encoders.tokenizer import SpecialTokens

SPECIAL_TOKEN_PIECES: Dict[str, str] = {
    "PAD": "<pad>",
    "BOS": "<s>",
    "EOS": "</s>",
    "UNK": "<unk>",
    "CLS": "<cls>",
    "SEP": "<sep>",
    "MARK": "<mark>",
    "MASK": "<mask>",
    "INSTRUCTION": "<instruction>",
    "END_INSTRUCTION": "</instruction>",
    "SYSTEM": "<system>",
    "END_SYSTEM": "</system>",
    "USER": "<user>",
    "END_USER": "</user>",
    "AGENT": "<agent>",
    "END_AGENT": "</agent>",
    "QUESTION": "<question>",
    "END_QUESTION": "</question>",
    "ANSWER": "<answer>",
    "END_ANSWER": "</answer>",
}

CORE_FIELDS = ("PAD", "BOS", "EOS", "UNK")


def control_symbols(special_tokens: SpecialTokens) -> List[str]:
    """
    Ordered pieces for every special token outside SentencePiece's own
    pad/bos/eos/unk ids, ascending by id.

    Parameters
    ----------
    special_tokens : SpecialTokens
        Special token id scheme.

    Returns
    -------
    list of str
        Piece strings, e.g. ``["<cls>", "<sep>", ...]``.
    """
    names = [name for name in SPECIAL_TOKEN_PIECES if name not in CORE_FIELDS]
    names.sort(key=lambda name: getattr(special_tokens, name))
    return [SPECIAL_TOKEN_PIECES[name] for name in names]


def verify_tokenizer_alignment(
    model_path: str, special_tokens: Optional[SpecialTokens] = None
) -> Dict[str, int]:
    """
    Check that every special token in a trained model landed at its
    declared id. SentencePiece assigns reserved ids by its own internal
    packing rule, not by the order symbols are requested in, so this is
    checked rather than assumed.

    Parameters
    ----------
    model_path : str
        Path to a trained SentencePiece `.model` file.
    special_tokens : SpecialTokens, optional
        Expected id scheme. Defaults to `SpecialTokens()`.

    Returns
    -------
    dict
        Special token name to its actual id in the model.

    Raises
    ------
    ValueError
        If any special token's actual id does not match `special_tokens`.
    """
    special_tokens = special_tokens or SpecialTokens()
    processor = spm.SentencePieceProcessor(model_file=model_path)

    actual_ids = {
        name: processor.piece_to_id(piece)
        for name, piece in SPECIAL_TOKEN_PIECES.items()
    }
    mismatches = [
        f"{name}: expected {getattr(special_tokens, name)}, got {actual_ids[name]}"
        for name in SPECIAL_TOKEN_PIECES
        if actual_ids[name] != getattr(special_tokens, name)
    ]
    if mismatches:
        raise ValueError(
            "special token ids drifted from SpecialTokens:\n" + "\n".join(mismatches)
        )
    return actual_ids


def fit_tokenizer(
    corpus_paths: List[str],
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    special_tokens: Optional[SpecialTokens] = None,
    **train_kwargs,
) -> str:
    """
    Train a SentencePiece model and write `<model_prefix>.model` / `.vocab`.

    Reserves ids for every field on `special_tokens`: PAD, BOS, EOS and UNK
    take SentencePiece's own core ids, the rest become control symbols in
    ascending id order, and ordinary vocabulary starts at
    `special_tokens.TOKEN_OFFSET`. The result loads directly with
    `SentencePieceTokenizer(model_path)`.

    Parameters
    ----------
    corpus_paths : list of str
        Text files or glob patterns to train on, one sample per line.
    model_prefix : str
        Output path prefix; produces `<model_prefix>.model` and `.vocab`.
    vocab_size : int, optional
        Total vocabulary size, special tokens included.
    model_type : str, optional
        SentencePiece model type: "bpe", "unigram", "char", or "word".
    special_tokens : SpecialTokens, optional
        Special token id scheme. Defaults to `SpecialTokens()`.
    **train_kwargs
        Extra keyword arguments forwarded to `spm.SentencePieceTrainer.train`.

    Returns
    -------
    str
        Path to the written `.model` file.

    Raises
    ------
    FileNotFoundError
        If no file matches `corpus_paths`.
    ValueError
        If the trained model's special token ids do not match
        `special_tokens` (see `verify_tokenizer_alignment`).
    """
    special_tokens = special_tokens or SpecialTokens()
    corpus_files = sorted(
        {path for pattern in corpus_paths for path in glob.glob(pattern)}
    )
    if not corpus_files:
        raise FileNotFoundError(f"no corpus files matched: {corpus_paths}")

    spm.SentencePieceTrainer.train(
        input=",".join(corpus_files),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=special_tokens.PAD,
        bos_id=special_tokens.BOS,
        eos_id=special_tokens.EOS,
        unk_id=special_tokens.UNK,
        control_symbols=control_symbols(special_tokens),
        **train_kwargs,
    )

    model_path = f"{model_prefix}.model"
    verify_tokenizer_alignment(model_path, special_tokens)
    return model_path
