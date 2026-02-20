# `stembed-rs`

A lightweight Rust library for hosting [static embedding models](https://huggingface.co/blog/static-embeddings) with included WASM bindings.

## Static Embedding Models

Static embedding models offer a highly efficient alternative to transformer-based models (like BERT) for semantic text embedding tasks. Unlike transformers that use attention mechanisms to compute context-aware representations, static models map each token in a vocabulary to a fixed vector. The resulting embedding is given as the mean of the per-token embeddings. While naive, this approach when distilled from the weights of large, pretrained models, performs competitively on text retrieval benchmarks.

## Supported Models

Pretrained weights for each model can be bundled into your Rust application using their corresponding feature flag. This allows you to skip writing logic for downloading model weights at the cost of a larger bundle size.

| Model | Dimensions | Feature Flag |
|-------|-----------|--------------|
| [Potion-Base-8M](https://huggingface.co/minishlab/potion-base-8M) | 256 | `potion-base-8m` |

