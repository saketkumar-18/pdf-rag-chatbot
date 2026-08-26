"""Generate a multi-page demo PDF used by tests and for trying the app out.

Run:  .venv/Scripts/python.exe scripts/make_demo_pdf.py
Output: data/demo/attention-is-all-you-need-summary.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpdf import FPDF  # noqa: E402

PAGES = [
    # page 1 ------------------------------------------------------------
    """Attention Is All You Need - Study Summary

Overview

This document is a study summary of the landmark 2017 paper "Attention Is All
You Need" by Vaswani et al. It introduces the Transformer, a neural sequence
model built entirely on attention mechanisms, removing recurrence entirely.

Motivation

Recurrent models such as LSTM and GRU process tokens one at a time, which
prevents parallelization across sequence positions during training. For long
sequences this becomes a serious bottleneck. The Transformer removes all
recurrence and instead relies on a mechanism called self-attention, which lets
every position attend to every other position in a single layer.""",
    # page 2 ------------------------------------------------------------
    """Architecture of the Transformer

Encoder Stack

The encoder is a stack of six identical layers. Each layer contains two
sub-layers: a multi-head self-attention mechanism and a position-wise fully
connected feed-forward network. Residual connections plus layer normalization
are applied around each sub-layer.

Decoder Stack

The decoder also has six layers but adds a third sub-layer which performs
multi-head attention over the output of the encoder stack. The decoder attends
to previously generated tokens with masking to preserve autoregression.

Scaled Dot-Product Attention

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. The scaling factor
sqrt(d_k) counteracts the growth of dot products for large dimensions, which
would otherwise push softmax into regions with tiny gradients.""",
    # page 3 ------------------------------------------------------------
    """Training Details and Results

Datasets

Experiments use two machine translation corpora: WMT 2014 English-German
(4.5 million sentence pairs) and WMT 2014 English-French (36 million pairs).
Sentences are encoded as byte-pair embeddings with a shared vocabulary of
about 37000 tokens.

Hardware and Schedule

Models train on eight NVIDIA P100 GPUs. The base model trains for 12 hours
(100000 steps) and the big model for 3.5 days (300000 steps).

Results

The base Transformer reaches 27.3 BLEU on English-to-German and 38.1 BLEU on
English-to-French, beating all prior single models at a fraction of the
training cost. The big model reaches 28.4 BLEU on English-German.""",
]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for text in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, text)
    out_path = out_dir / "attention-is-all-you-need-summary.pdf"
    pdf.output(str(out_path))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
