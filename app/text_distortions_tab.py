"""Text distortions tab: TextProcessor's masking, deletion, infilling and pairing tasks, one section per task."""

import html

import streamlit as st
from common import run_panel
from polyergalio.encoders.text_encoders import DistortionTask, TextProcessor

HIGHLIGHT_STYLE = "background:#ffe08a; color:#1a1a1a; padding:0.1em 0.3em; border-radius:0.35em; font-weight:600;"

try:
    import text_distortions_example as distortions
except ImportError as error:
    distortions = None
    import_error = error

TASK_NAMES = {task.value: task for task in DistortionTask}


def run(
    corpus: list,
    task_names: list,
    vocab_size: int,
    max_length: int,
    mask_prob: float,
    replace_prob: float,
    span_geometric_p: float,
    max_span_length: int,
    delete_prob: float,
    infill_prob: float,
    infill_poisson_lambda: float,
    seed: int,
    num_shown: int,
) -> None:
    """Fit a tokenizer and TextProcessor on the corpus, then stash each task's distorted lines for render() to lay out."""
    distortions.VOCAB_SIZE = vocab_size
    tokenizer = distortions.fit_text_tokenizer(corpus)
    tasks = [TASK_NAMES[name] for name in task_names]
    processor = TextProcessor(
        tokenizer,
        max_length=max_length,
        tasks=tasks,
        mask_prob=mask_prob,
        replace_prob=replace_prob,
        span_geometric_p=span_geometric_p,
        max_span_length=max_span_length,
        delete_prob=delete_prob,
        infill_prob=infill_prob,
        infill_poisson_lambda=infill_poisson_lambda,
        random_seed=seed,
    )
    processor.fit(corpus)
    print(f"fitted a {tokenizer.get_vocab_size()}-token vocabulary on {len(corpus)} lines")

    sentences = corpus[:num_shown]
    st.session_state["td_data"] = dict(
        original=[distortions.original_line(processor, tokenizer, text) for text in sentences],
        by_task={
            task.value: [distortions.distorted_tokens(processor, tokenizer, task, text) for text in sentences]
            for task in tasks
        },
    )


def render_highlighted(tokens: list) -> str:
    """Inline HTML for one sentence: target-masked tokens get a highlight, everything else is plain text."""
    pieces = [
        f'<mark style="{HIGHLIGHT_STYLE}">{html.escape(piece)}</mark>' if is_target else html.escape(piece)
        for piece, is_target in tokens
    ]
    return " ".join(pieces)


def show_plain_lines(lines: list) -> None:
    """Numbered plain-text lines, no highlighting."""
    for i, line in enumerate(lines, start=1):
        st.markdown(f"**{i}.** {html.escape(line)}")


def show_token_lines(rows: list) -> None:
    """Numbered lines with target-masked tokens highlighted."""
    for i, tokens in enumerate(rows, start=1):
        st.markdown(f"**{i}.** {render_highlighted(tokens)}", unsafe_allow_html=True)


def render() -> None:
    if distortions is None:
        st.error(f"The text distortions example needs its dependencies: {import_error}")
        return

    left, right = st.columns([1, 2])
    with left:
        st.write(
            "TextProcessor builds corrupted training views of text for pretraining an encoder: masking, "
            "replacing, deleting and infilling tokens, plus a next-sentence pairing task."
        )
        st.subheader("Settings")
        tasks = st.multiselect(
            "Tasks",
            list(TASK_NAMES),
            default=[task.value for task in distortions.TASKS],
            key="td_tasks",
        )
        vocab_size = int(st.number_input("Vocab size", 50, 2000, distortions.VOCAB_SIZE, step=50, key="td_vocab"))
        max_length = int(st.number_input("Max length", 8, 256, distortions.MAX_LENGTH, step=8, key="td_maxlen"))
        num_shown = int(st.number_input("Sentences shown per task", 1, 10, distortions.NUM_SHOWN, key="td_shown"))
        seed = int(st.number_input("Seed", 0, 9999, distortions.RANDOM_SEED, key="td_seed"))
        st.subheader("CLOZE / SPAN_BOUNDARY / REPLACED_TOKEN")
        mask_prob = float(st.number_input("Mask probability", 0.0, 1.0, 0.15, step=0.05, key="td_mask_prob"))
        replace_prob = float(st.number_input("Replace probability", 0.0, 1.0, 0.15, step=0.05, key="td_replace_prob"))
        span_geometric_p = float(st.number_input("Span geometric p", 0.05, 1.0, 0.2, step=0.05, key="td_span_p"))
        max_span_length = int(st.number_input("Max span length", 1, 30, 10, key="td_span_len"))
        st.subheader("TOKEN_DELETION / TEXT_INFILLING")
        delete_prob = float(st.number_input("Delete probability", 0.0, 1.0, 0.15, step=0.05, key="td_delete_prob"))
        infill_prob = float(st.number_input("Infill probability", 0.0, 1.0, 0.3, step=0.05, key="td_infill_prob"))
        infill_poisson_lambda = float(
            st.number_input("Infill span length (Poisson mean)", 0.5, 20.0, 3.0, step=0.5, key="td_infill_lambda")
        )
        st.subheader("Data")
        corpus_text = st.text_area(
            "Corpus, one sentence (or short passage) per line",
            value="\n".join(dict.fromkeys(distortions.CORPUS)),
            height=160,
            key="td_corpus",
        )
    if not tasks:
        right.warning("Choose at least one task.")
        return

    corpus = [line.strip() for line in corpus_text.splitlines() if line.strip()]
    if not corpus:
        right.warning("The corpus needs at least one line.")
        return

    with right:
        run_panel(
            "td",
            run,
            corpus,
            tasks,
            vocab_size,
            max_length,
            mask_prob,
            replace_prob,
            span_geometric_p,
            max_span_length,
            delete_prob,
            infill_prob,
            infill_poisson_lambda,
            seed,
            num_shown,
        )
        if "td_data" in st.session_state:
            data = st.session_state["td_data"]
            st.subheader("Original")
            show_plain_lines(data["original"])
            for task_name, rows in data["by_task"].items():
                with st.expander(task_name, expanded=True):
                    show_token_lines(rows)
