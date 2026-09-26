"""Tokenizer module using SentencePiece with special token management."""

import glob
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm

from polyergalio.models.constants import DECISION_TYPES

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

@dataclass
class SpecialTokens:
    """Special token identifiers and their values."""

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2
    CLS: int = 3
    SEP: int = 4
    MARK: int = 5
    UNK: int = 6
    MASK: int = 7
    INSTRUCTION: int = 8
    END_INSTRUCTION: int = 9
    SYSTEM: int = 10
    END_SYSTEM: int = 11
    USER: int = 12
    END_USER: int = 13
    AGENT: int = 14
    END_AGENT: int = 15
    QUESTION: int = 16
    END_QUESTION: int = 17
    ANSWER: int = 18
    END_ANSWER: int = 19
    TOKEN_OFFSET: int = 20


class SentencePieceTokenizer:
    """
    SentencePiece-based tokenizer with predefined special tokens.

    Parameters
    ----------
    model_path : str
        Path to the SentencePiece model file (.model).
    special_tokens : SpecialTokens, optional
        Special token configuration. Defaults to SpecialTokens().

    Raises
    ------
    ImportError
        If SentencePiece is not installed.
    """

    def __init__(self, model_path: str, special_tokens: Optional[SpecialTokens] = None):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.special_tokens = special_tokens or SpecialTokens()
        self._validate_special_tokens()

    def _validate_special_tokens(self) -> None:
        """Verify special token ids are within vocab bounds."""
        vocab_size = self.sp.get_piece_size()
        max_special = max(
            self.special_tokens.PAD,
            self.special_tokens.BOS,
            self.special_tokens.EOS,
            self.special_tokens.CLS,
            self.special_tokens.SEP,
            self.special_tokens.MARK,
            self.special_tokens.UNK,
            self.special_tokens.MASK,
            self.special_tokens.INSTRUCTION,
            self.special_tokens.END_INSTRUCTION,
            self.special_tokens.SYSTEM,
            self.special_tokens.END_SYSTEM,
            self.special_tokens.USER,
            self.special_tokens.END_USER,
            self.special_tokens.QUESTION,
            self.special_tokens.END_QUESTION,
            self.special_tokens.ANSWER,
            self.special_tokens.END_ANSWER,
        )
        if max_special >= vocab_size:
            raise ValueError(
                f"Special token ids exceed vocab size. "
                f"Max special: {max_special}, vocab size: {vocab_size}"
            )

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token ids.

        Parameters
        ----------
        text : str
            Input text.
        add_bos : bool, optional
            Prepend BOS token.
        add_eos : bool, optional
            Append EOS token.

        Returns
        -------
        List[int]
            Token ids.
        """
        ids = self.sp.encode(text, out_type=int)

        if add_bos:
            ids = [self.special_tokens.BOS] + ids
        if add_eos:
            ids = ids + [self.special_tokens.EOS]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token ids to text.

        Parameters
        ----------
        ids : List[int]
            Token ids.
        skip_special : bool, optional
            Skip special tokens during decoding.

        Returns
        -------
        str
            Decoded text.
        """
        if skip_special:
            ids = self._filter_special_tokens(ids)

        return self.sp.decode(ids)

    def _filter_special_tokens(self, ids: List[int]) -> List[int]:
        """Remove special token ids from sequence."""
        special_ids = {
            self.special_tokens.PAD,
            self.special_tokens.BOS,
            self.special_tokens.EOS,
            self.special_tokens.CLS,
            self.special_tokens.SEP,
            self.special_tokens.MARK,
            self.special_tokens.UNK,
            self.special_tokens.MASK,
            self.special_tokens.INSTRUCTION,
            self.special_tokens.END_INSTRUCTION,
            self.special_tokens.SYSTEM,
            self.special_tokens.END_SYSTEM,
            self.special_tokens.USER,
            self.special_tokens.END_USER,
            self.special_tokens.QUESTION,
            self.special_tokens.END_QUESTION,
            self.special_tokens.ANSWER,
            self.special_tokens.END_ANSWER,
        }
        return [tid for tid in ids if tid not in special_ids]

    def get_vocab_size(self) -> int:
        """Return total vocabulary size."""
        return self.sp.get_piece_size()

    def id_to_piece(self, token_id: int) -> str:
        """
        The vocabulary piece for one token id, special tokens included as
        their literal piece string (e.g. "<mask>"), unlike decode().
        """
        return self.sp.id_to_piece(int(token_id))

    def is_special_token(self, token_id: int) -> bool:
        """Check if token id is special."""
        special_ids = {
            self.special_tokens.PAD,
            self.special_tokens.BOS,
            self.special_tokens.EOS,
            self.special_tokens.CLS,
            self.special_tokens.SEP,
            self.special_tokens.MARK,
            self.special_tokens.UNK,
            self.special_tokens.MASK,
            self.special_tokens.INSTRUCTION,
            self.special_tokens.END_INSTRUCTION,
            self.special_tokens.SYSTEM,
            self.special_tokens.END_SYSTEM,
            self.special_tokens.USER,
            self.special_tokens.END_USER,
            self.special_tokens.QUESTION,
            self.special_tokens.END_QUESTION,
            self.special_tokens.ANSWER,
            self.special_tokens.END_ANSWER,
        }
        return token_id in special_ids

    def get_special_token_id(self, name: str) -> int:
        """
        Get special token id by name.

        Parameters
        ----------
        name : str
            Token name (e.g., 'BOS', 'EOS', 'PAD').

        Returns
        -------
        int
            Token id.

        Raises
        ------
        ValueError
            If name is not a known special token.
        """
        name_upper = name.upper()
        if hasattr(self.special_tokens, name_upper):
            return getattr(self.special_tokens, name_upper)
        raise ValueError(f"Unknown special token: {name}")


class TokenSequenceBuilder:
    """Build token sequences with structural patterns."""

    def __init__(self, tokenizer: "SentencePieceTokenizer"):
        self.tokenizer = tokenizer
        self.st = tokenizer.special_tokens

    def build_cls_sequence(
        self,
        text: str,
        add_sep: bool = True,
    ) -> List[int]:
        """
        Build sequence: [CLS] <text> [SEP].

        Parameters
        ----------
        text : str
            Input text.
        add_sep : bool, optional
            Append SEP token.

        Returns
        -------
        List[int]
            Token sequence.
        """
        tokens = [self.st.CLS] + self.tokenizer.encode(text)
        if add_sep:
            tokens.append(self.st.SEP)
        return tokens

    def build_paired_sequence(
        self,
        text_a: str,
        text_b: str,
    ) -> List[int]:
        """
        Build sequence: [CLS] <text_a> [SEP] <text_b> [SEP].

        Parameters
        ----------
        text_a : str
            First text.
        text_b : str
            Second text.

        Returns
        -------
        List[int]
            Token sequence.
        """
        tokens = (
            [self.st.CLS]
            + self.tokenizer.encode(text_a)
            + [self.st.SEP]
            + self.tokenizer.encode(text_b)
            + [self.st.SEP]
        )
        return tokens

    def build_marked_options_sequence(
        self,
        header: str,
        options: List[str],
        add_seps: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Build sequence: [CLS] <header> [SEP] [MARK] <opt0> [MARK] <opt1> ...

        Parameters
        ----------
        header : str
            Header text.
        options : List[str]
            Option texts.
        add_seps : bool, optional
            Append final SEP token.

        Returns
        -------
        Tuple[List[int], List[int]]
            (token sequence, marker positions).
        """
        tokens = [self.st.CLS] + self.tokenizer.encode(header) + [self.st.SEP]
        marker_positions = []

        for opt in options:
            marker_positions.append(len(tokens))
            tokens.append(self.st.MARK)
            tokens.extend(self.tokenizer.encode(opt))

        if add_seps:
            tokens.append(self.st.SEP)

        return tokens, marker_positions

    def build_decision_sequence(
        self,
        instructions: str,
        options: List[str],
        state,
        decisiontype: DECISION_TYPES = DECISION_TYPES.CHOICE,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        # This is the decision support 
        Build sequence: [CLS] <type> <instructions> [SEP] [MARK] opt0 [MARK] opt1 ... [SEP] <state> [SEP].

        Parameters
        ----------
        instructions : str
            Question text.
        options : List[str]
            Option texts, in answer-index order.
        state : str or object
            Context the question is about; non-strings are JSON-serialized.
        decisiontype : DECISION_TYPES, optional
            Question type, prepended to `instructions` as its lowercase name.
        max_length : int, optional
            Truncate the state so the sequence fits; unlimited if None.

        Returns
        -------
        Tuple[List[int], List[int]]
            (token sequence, position of each option's MARK token).
        """
        header = self.tokenizer.encode(f"{decisiontype.name.lower()} {instructions}")
        state_text = state if isinstance(state, str) else json.dumps(state)
        state_ids = self.tokenizer.encode(state_text)

        tokens = [self.st.CLS] + header + [self.st.SEP]
        marker_positions = []
        for option in options:
            marker_positions.append(len(tokens))
            tokens.append(self.st.MARK)
            tokens.extend(self.tokenizer.encode(option))
        tokens.append(self.st.SEP)

        room = len(state_ids) if max_length is None else max(0, max_length - len(tokens) - 1)
        tokens.extend(state_ids[:room])
        tokens.append(self.st.SEP)
        return tokens, marker_positions

    def build_binary_decision(
        self, instructions: str, state, max_length: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Build a BINARY decision sequence: options fixed as ["false", "true"],
        matching `decode_decisions`' P(true) convention.

        Parameters
        ----------
        instructions : str
        state : str or object
        max_length : int, optional

        Returns
        -------
        Tuple[List[int], List[int]]
        """
        return self.build_decision_sequence(
            instructions, ["false", "true"], state,
            decisiontype=DECISION_TYPES.BINARY, max_length=max_length,
        )

    def build_choice_decision(
        self, instructions: str, options: List[str], state, max_length: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Build a CHOICE decision sequence over arbitrary options.

        Parameters
        ----------
        instructions : str
        options : List[str]
        state : str or object
        max_length : int, optional

        Returns
        -------
        Tuple[List[int], List[int]]
        """
        return self.build_decision_sequence(
            instructions, options, state,
            decisiontype=DECISION_TYPES.CHOICE, max_length=max_length,
        )

    def build_score_decision(
        self, instructions: str, num_levels: int, state, max_length: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Build a SCORE decision sequence: options are the ordered levels
        "0".."num_levels - 1", matching `decode_decisions`' expected-level
        convention.

        Parameters
        ----------
        instructions : str
        num_levels : int
        state : str or object
        max_length : int, optional

        Returns
        -------
        Tuple[List[int], List[int]]
        """
        levels = [str(level) for level in range(num_levels)]
        return self.build_decision_sequence(
            instructions, levels, state,
            decisiontype=DECISION_TYPES.SCORE, max_length=max_length,
        )

    def create_padding_mask(self, ids: List[int]) -> List[bool]:
        """
        Create binary mask for non-padding tokens.

        Parameters
        ----------
        ids : List[int]
            Token sequence.

        Returns
        -------
        List[bool]
            True for real tokens, False for padding.
        """
        return [tid != self.st.PAD for tid in ids]

    def build_instruction_sequence(
        self,
        instruction: str,
        content: str,
    ) -> List[int]:
        """
        Build sequence: [INSTRUCTION] <instruction> [/INSTRUCTION] <content>.

        Parameters
        ----------
        instruction : str
            Instruction text.
        content : str
            Content to process.

        Returns
        -------
        List[int]
            Token sequence.
        """
        tokens = (
            [self.st.INSTRUCTION]
            + self.tokenizer.encode(instruction)
            + [self.st.END_INSTRUCTION]
            + self.tokenizer.encode(content)
        )
        return tokens

    def build_system_user_sequence(
        self,
        system: str,
        user_input: str,
    ) -> List[int]:
        """
        Build sequence: [SYSTEM] <system> [/SYSTEM] [USER] <user_input> [/USER].

        Parameters
        ----------
        system : str
            System prompt.
        user_input : str
            User input text.

        Returns
        -------
        List[int]
            Token sequence.
        """
        tokens = (
            [self.st.SYSTEM]
            + self.tokenizer.encode(system)
            + [self.st.END_SYSTEM]
            + [self.st.USER]
            + self.tokenizer.encode(user_input)
            + [self.st.END_USER]
        )
        return tokens

    def build_qa_sequence(
        self,
        question: str,
        answer: str,
    ) -> List[int]:
        """
        Build sequence: [QUESTION] <question> [/QUESTION] [ANSWER] <answer> [/ANSWER].

        Parameters
        ----------
        question : str
            Question text.
        answer : str
            Answer text.

        Returns
        -------
        List[int]
            Token sequence.
        """
        tokens = (
            [self.st.QUESTION]
            + self.tokenizer.encode(question)
            + [self.st.END_QUESTION]
            + [self.st.ANSWER]
            + self.tokenizer.encode(answer)
            + [self.st.END_ANSWER]
        )
        return tokens

    def build_multi_turn_sequence(
        self,
        system: Optional[str] = None,
        turns: Optional[List[Tuple[str, str]]] = None,
    ) -> List[int]:
        """
        Build multi-turn conversation: [SYSTEM] ... [/SYSTEM] [USER] ... [/USER] [ANSWER] ... [/ANSWER] ...

        Parameters
        ----------
        system : str, optional
            System prompt.
        turns : List[Tuple[str, str]], optional
            List of (user_input, assistant_response) tuples.

        Returns
        -------
        List[int]
            Token sequence.
        """
        tokens = []

        if system:
            tokens.extend(
                [self.st.SYSTEM] + self.tokenizer.encode(system) + [self.st.END_SYSTEM]
            )

        if turns:
            for user_input, response in turns:
                tokens.extend(
                    [self.st.USER]
                    + self.tokenizer.encode(user_input)
                    + [self.st.END_USER]
                    + [self.st.ANSWER]
                    + self.tokenizer.encode(response)
                    + [self.st.END_ANSWER]
                )

        return tokens


def pad_sequences(
    sequences: List[List[int]],
    max_length: Optional[int] = None,
    pad_token: int = 0,
    pad_direction: str = "right",
) -> Tuple[List[List[int]], List[List[bool]]]:
    """
    Pad sequences to uniform length.

    Parameters
    ----------
    sequences : List[List[int]]
        Token sequences.
    max_length : int, optional
        Target length. Defaults to max sequence length.
    pad_token : int, optional
        Padding token id.
    pad_direction : str, optional
        'right' or 'left' padding.

    Returns
    -------
    Tuple[List[List[int]], List[List[bool]]]
        (padded sequences, attention masks).
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    masks = []

    for seq in sequences:
        length = len(seq)
        if length < max_length:
            pad_len = max_length - length
            if pad_direction == "right":
                padded_seq = seq + [pad_token] * pad_len
            else:
                padded_seq = [pad_token] * pad_len + seq
        else:
            padded_seq = seq[:max_length]

        padded.append(padded_seq)
        masks.append(
            [
                1 if (i < length or pad_direction == "left") else 0
                for i in range(max_length)
            ]
        )

    return padded, masks


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
