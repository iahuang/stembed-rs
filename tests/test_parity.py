#!/usr/bin/env python3
"""
Test harness: verify that stembed-rs produces identical embeddings to
HuggingFace sentence_transformers for static embedding models.

Usage:
    pip install sentence-transformers numpy
    python tests/test_parity.py

Requires: Rust toolchain (cargo)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUST_BINARY = PROJECT_ROOT / "target" / "release" / "test_embed"

MODELS = [
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
    "minishlab/potion-base-32M",
    "minishlab/potion-retrieval-32M",
    "joshcx/static-embedding-all-MiniLM-L6-v2",
    "joshcx/static-embedding-all-mpnet-base-v2",
]

# This string exercises many tokenizer edge cases:
#   - Accented characters (café, naïve, résumé, über, crème, brûlées)
#     → tests NFD decomposition + accent stripping
#   - CJK characters (你好世界) → tests Chinese character spacing
#   - Emoji (🌍) → tests handling of characters outside BMP / unknown tokens
#   - Possessives and contractions (café's, isn't)
#     → tests apostrophe as punctuation boundary
#   - Numbers and decimals (42, 3.14) → tests digit tokenization
#   - Various punctuation (: ! $ ?) → tests punctuation splitting
#   - Hyphenated compound (über-crème) → tests hyphen as word boundary
#   - Multiple consecutive spaces and a tab → tests whitespace normalization
#   - Mixed case (The) → tests lowercasing
TEST_TEXT = (
    "The café's naïve résumé: 42 über-crème brûlées cost $3.14!"
    " 你好世界 🌍 isn't   that\tweird?"
)

# Absolute tolerance for embedding comparison.
# Both implementations load the same safetensors weights and should produce
# nearly identical results. Differences come from:
#   - F16→F32 conversion (bit-exact on both sides)
#   - Float accumulation order (Rust loop vs PyTorch BLAS)
#   - Normalization epsilon (Rust 1e-8 vs PyTorch 1e-12)
ATOL = 1e-5


def build_rust():
    """Build the Rust test binary in release mode."""
    print("Building stembed test binary...")
    result = subprocess.run(
        [
            "cargo", "build",
            "-p", "stembed",
            "--features", "huggingface",
            "--release",
            "--bin", "test_embed",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("CARGO BUILD FAILED", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    print(f"  Binary: {RUST_BINARY}")


def get_rust_result(model_id: str, text: str) -> dict:
    """Run the Rust binary and return parsed JSON output."""
    result = subprocess.run(
        [str(RUST_BINARY), model_id, text],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Rust binary failed for {model_id}:\n{result.stderr}"
        )
    if result.stderr:
        # Print Rust diagnostics (model info, token counts)
        for line in result.stderr.strip().split("\n"):
            print(f"  [rust] {line}")
    return json.loads(result.stdout)


def get_python_result(model_id: str, text: str) -> dict:
    """Compute embedding using sentence_transformers and return result dict."""
    import torch
    from sentence_transformers import SentenceTransformer

    # Force CPU — MPS doesn't support embedding_bag
    model = SentenceTransformer(model_id, device=torch.device("cpu"))

    # Get the embedding (normalized, matching Rust behavior)
    embedding = model.encode(text, normalize_embeddings=True)

    # Extract token IDs from the StaticEmbedding module for diagnostics
    tokens = None
    try:
        static_module = model[0]
        encoded = static_module.tokenizer.encode(text, add_special_tokens=False)
        tokens = encoded.ids
    except Exception:
        pass  # Token extraction is best-effort for diagnostics

    dim = embedding.shape[0]
    print(f"  [python] dim={dim}", end="")
    if tokens is not None:
        print(f", tokens ({len(tokens)} total): {tokens}", end="")
    print()

    return {
        "embedding": embedding,
        "tokens": tokens,
        "dim": dim,
    }


def compare(model_id: str, text: str) -> bool:
    """Compare Rust and Python embeddings for a single model. Returns True if passed."""
    print(f"\n{'=' * 70}")
    print(f"  Model: {model_id}")
    print(f"  Text:  {text!r}")
    print(f"{'=' * 70}")

    try:
        py = get_python_result(model_id, text)
    except Exception as e:
        print(f"  SKIP (Python failed): {e}")
        return None  # Neither pass nor fail

    try:
        rs = get_rust_result(model_id, text)
    except Exception as e:
        print(f"  FAIL (Rust failed): {e}")
        return False

    py_emb = np.asarray(py["embedding"], dtype=np.float32)
    rs_emb = np.asarray(rs["embedding"], dtype=np.float32)

    # --- Dimension check ---
    if py_emb.shape != rs_emb.shape:
        print(f"  FAIL: dimension mismatch: Python {py_emb.shape} vs Rust {rs_emb.shape}")
        return False

    # --- Token comparison (diagnostic only) ---
    py_tokens = py.get("tokens")
    rs_tokens = rs.get("tokens")
    if py_tokens is not None and rs_tokens is not None:
        if list(py_tokens) == list(rs_tokens):
            print(f"  Tokens: MATCH ({len(rs_tokens)} tokens)")
        else:
            print(f"  Tokens: MISMATCH")
            print(f"    Python ({len(py_tokens)}): {py_tokens}")
            print(f"    Rust   ({len(rs_tokens)}): {rs_tokens}")
            # Find first divergence
            for i, (p, r) in enumerate(zip(py_tokens, rs_tokens)):
                if p != r:
                    print(f"    First divergence at index {i}: Python={p} vs Rust={r}")
                    break
            if len(py_tokens) != len(rs_tokens):
                print(f"    Length differs: Python={len(py_tokens)} vs Rust={len(rs_tokens)}")

    # --- Embedding comparison ---
    max_abs_diff = np.max(np.abs(py_emb - rs_emb))
    mean_abs_diff = np.mean(np.abs(py_emb - rs_emb))
    cosine_sim = np.dot(py_emb, rs_emb) / (
        np.linalg.norm(py_emb) * np.linalg.norm(rs_emb) + 1e-12
    )

    # L2 norms (both should be ~1.0 if normalized)
    py_norm = np.linalg.norm(py_emb)
    rs_norm = np.linalg.norm(rs_emb)

    print(f"  Norms:       Python={py_norm:.8f}  Rust={rs_norm:.8f}")
    print(f"  Cosine sim:  {cosine_sim:.10f}")
    print(f"  Max |diff|:  {max_abs_diff:.2e}")
    print(f"  Mean |diff|: {mean_abs_diff:.2e}")

    if max_abs_diff <= ATOL:
        print(f"  PASS (atol={ATOL})")
        return True
    else:
        print(f"  FAIL: max absolute difference {max_abs_diff:.2e} exceeds tolerance {ATOL}")
        # Show the indices with largest differences
        worst = np.argsort(np.abs(py_emb - rs_emb))[-5:][::-1]
        for idx in worst:
            print(f"    dim[{idx}]: Python={py_emb[idx]:.8e}  Rust={rs_emb[idx]:.8e}  diff={py_emb[idx]-rs_emb[idx]:.8e}")
        return False


def main():
    build_rust()

    results = {}
    for model_id in MODELS:
        result = compare(model_id, TEST_TEXT)
        results[model_id] = result

    # --- Summary ---
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'=' * 70}")
    for model_id, result in results.items():
        status = "PASS" if result is True else "FAIL" if result is False else "SKIP"
        short_name = model_id.split("/")[-1]
        print(f"  [{status}] {short_name}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
