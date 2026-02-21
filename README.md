# `stembed-rs`

A lightweight Rust library for hosting [static embedding models](https://huggingface.co/blog/static-embeddings) with included WASM bindings. This library is minimal, avoiding dependencies on ONNX runtime or other general-purpose machine learning frameworks.

## Static Embedding Models

Static embedding models offer a highly efficient alternative to transformer-based models (like BERT) for semantic text embedding tasks. Unlike transformers that use attention mechanisms to compute context-aware representations, static models map each token in a vocabulary to a fixed vector. The resulting embedding is given as the mean of the per-token embeddings. While naive, this approach when distilled from the weights of large, pretrained models, performs competitively on text retrieval benchmarks.

## Supported Models

Potion-Base-8M has a feature flag to bundle the model weights (15MB) into your application, skipping the need to download the weights at runtime. Using the `huggingface` feature flag, additional models can be downloaded from Hugging Face at runtime.

| Model | Dimensions | Feature Flag |
|-------|-----------|--------------|
| [Potion-Base-8M](https://huggingface.co/minishlab/potion-base-8M) | 256 | `potion-base-8m` |
| [Sentence Transformers Static Retrieval English](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | 1024 | N/A |
| [Sentence Transformers Static Similarity Multilingual](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | 1024 | N/A |