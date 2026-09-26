"""Spectre encoder-decoder tab: masked language modelling, decoder training, generation."""

import numpy as np
import pandas as pd
import streamlit as st
from common import run_panel, show_diagram
from polyergalio.encoders.text_encoders import TextProcessor
from polyergalio.visuals.nnet_visuals import plot_network

try:
    import NNet_spectre_example as spectre
except ImportError as error:
    spectre = None
    import_error = error

CATEGORIES = ("determiners", "adjectives", "nouns", "verbs", "adverbs")
ATTRIBUTES = ("DETERMINERS", "ADJECTIVES", "NOUNS", "VERBS", "ADVERBS")


def configure(hidden: int, heads: int, mlm_steps: int, decoder_steps: int, words: dict) -> None:
    """Point the example module's constants at the chosen settings."""
    spectre.HIDDEN_DIM = hidden
    spectre.NUM_HEADS = heads
    spectre.MLM_STEPS = mlm_steps
    spectre.DECODER_STEPS = decoder_steps
    for attribute, values in words.items():
        setattr(spectre, attribute, values)


def parse_words(text: str) -> tuple:
    """Comma separated words as a tuple."""
    return tuple(word.strip() for word in text.split(",") if word.strip())


def train(extra: list, seed: int) -> None:
    """Run the example's three stages and keep the trained networks for reconstruction."""
    rng = np.random.default_rng(seed)
    train_texts, held_out = spectre.generate_corpus(rng)
    train_texts = train_texts + extra
    print(f"{len(train_texts)} training sentences, {len(held_out)} held out")

    tokenizer = spectre.fit_text_tokenizer(train_texts)
    vocab_size = tokenizer.get_vocab_size()
    processor = TextProcessor(
        tokenizer,
        max_length=spectre.SEQUENCE_LENGTH,
        tasks=("cloze",),
        mask_prob=spectre.MASK_PROB,
        random_seed=seed,
    )
    processor.fit(train_texts)

    encoder = spectre.build_encoder(vocab_size, padding_idx=tokenizer.special_tokens.PAD)
    print("--- stage 1: encoder, masked language modelling ---")
    spectre.train_mlm(encoder, processor, train_texts, held_out, rng)

    seq2seq = spectre.build_seq2seq(encoder, vocab_size)
    print("--- stage 2: decoder training ---")
    spectre.train_decoder(seq2seq, processor, train_texts, held_out, rng)

    print("--- stage 3: generation from held-out sentences ---")
    shown = held_out[: spectre.NUM_SHOWN]
    result = spectre.generate(encoder, seq2seq, processor, shown)
    spectre.show_generation(processor, shown, result)
    print(f"cached decoding vs full causal forward, max logit difference: {result['cache_difference']:.2e}")

    st.session_state["spectre_models"] = dict(encoder=encoder, seq2seq=seq2seq, processor=processor)


def reconstruct(text: str) -> tuple[str, str]:
    """
    Encode one sentence with the trained encoder and decode it from its sentence vector.

    Returns
    -------
    the generated sentence, and its token by token (piece, probability) trace
    """
    models = st.session_state["spectre_models"]
    tokenizer = models["processor"].tokenizer
    result = spectre.generate(models["encoder"], models["seq2seq"], models["processor"], [text])
    length = int(result["lengths"][0])
    logits = result["logits"][0]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    lines = ["token by token (piece, probability):"]
    for position in range(length):
        token = int(result["tokens"][0, position])
        piece = tokenizer.decode([token], skip_special=False)
        lines.append(f"  {position:2d}  {piece:<8s} {probabilities[position, token]:.3f}")
    return tokenizer.decode(result["tokens"][0, :length].tolist()), "\n".join(lines)


def render() -> None:
    if spectre is None:
        st.error(f"The Spectre example needs its dependencies: {import_error}")
        return
    left, right = st.columns([1, 2])
    with left:
        st.write(
            "A Spectre encoder learns masked language modelling on a small grammar, its pooled sentence vector "
            "conditions a causal Spectre decoder, and sentences are regenerated token by token from that vector."
        )
        st.subheader("Settings")
        hidden = int(st.number_input("Hidden dim", 8, 256, spectre.HIDDEN_DIM, step=8, key="sp_hidden"))
        heads = int(st.number_input("Heads", 1, 16, spectre.NUM_HEADS, key="sp_heads"))
        mlm_steps = int(st.number_input("MLM steps", 1, 5000, spectre.MLM_STEPS, step=50, key="sp_mlm"))
        decoder_steps = int(st.number_input("Decoder steps", 1, 5000, spectre.DECODER_STEPS, step=50, key="sp_decoder"))
        seed = int(st.number_input("Seed", 0, 9999, 0, key="sp_seed"))
        st.subheader("Data settings")
        extra_text = st.text_area("Extra training sentences, one per line", key="sp_extra")
    if hidden % heads:
        right.warning("Hidden dim must be divisible by the head count.")
        return

    with right:
        st.subheader("Data")
        defaults = {name: ", ".join(getattr(spectre, name)) for name in ATTRIBUTES}
        table = st.data_editor(
            pd.DataFrame({"category": CATEGORIES, "words": [defaults[name] for name in ATTRIBUTES]}),
            disabled=["category"],
            num_rows="fixed",
            key="sp_words",
        )
        st.caption("Sentences are built as: determiner adjective noun verb adverb. Edit the word lists to change the grammar.")
        words = {attribute: parse_words(text) for attribute, text in zip(ATTRIBUTES, table["words"])}
        extra = [line.strip() for line in extra_text.splitlines() if line.strip()]

        configure(hidden, heads, mlm_steps, decoder_steps, words)
        with st.expander("Model structure", expanded=True):
            encoder = spectre.build_encoder(spectre.TARGET_VOCAB_SIZE, padding_idx=0)
            seq2seq = spectre.build_seq2seq(encoder, spectre.TARGET_VOCAB_SIZE)
            first, second = st.columns(2)
            with first:
                show_diagram(plot_network(encoder, figsize=(5, 9)).figure, width=260)
            with second:
                show_diagram(plot_network(seq2seq, figsize=(8, 16)).figure, width=340)

        st.caption("Training runs in NumPy and can take several minutes at the default step counts.")
        run_panel("sp", train, extra, seed, label="Train and generate")

        if "spectre_models" in st.session_state:
            text = st.text_input("Reconstruct a sentence with the trained model", "the red cat runs fast.", key="sp_text")
            if st.button("Reconstruct", key="sp_reconstruct"):
                sentence, trace = reconstruct(text)
                st.code(f"{sentence}\n\n{trace}", language="text")
