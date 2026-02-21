# `stembed-rs`

A lightweight Rust library for hosting [static embedding models](https://huggingface.co/blog/static-embeddings) with included WASM bindings. This library is minimal, avoiding dependencies on more complex, general-purpose machine learning frameworks such as ONNX runtime.

## Static Embedding Models

Static embedding models offer a highly efficient alternative to transformer-based models (like BERT) for semantic text embedding tasks. Unlike transformers that use attention mechanisms to compute context-aware representations, static models map each token in a vocabulary to a fixed vector. The resulting embedding is given as the mean of the per-token embeddings. While naive, this approach when distilled from the weights of large, pretrained models, performs competitively on text retrieval benchmarks.

## Supported Models

Potion-Base-8M has a feature flag `potion-base-8m` to bundle the model weights (15MB, FP16) into your application, skipping the need to download the weights at runtime. This allows the use of the `pretrained_potion_base_8m` function to instantiate a loaded model.

Using the `huggingface` feature flag, additional models can be downloaded from Hugging Face at runtime. The following models are supported:

| Model (HuggingFace) | Dimensions |
|-------|-----------|
| [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) | 256 |
| [`minishlab/potion-base-32M`](https://huggingface.co/minishlab/potion-base-32M) | 256 |
| [`minishlab/potion-retrieval-32M`](https://huggingface.co/minishlab/potion-retrieval-32M) | 256 |
| [`sentence-transformers/static-retrieval-mrl-en-v1`](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) | 1024 |
| [`sentence-transformers/static-similarity-mrl-multilingual-v1`](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) | 1024 |
| [`joshcx/static-embedding-all-MiniLM-L6-v2`](https://huggingface.co/joshcx/static-embedding-all-MiniLM-L6-v2) | 256 |
| [`joshcx/static-embedding-all-mpnet-base-v2`](https://huggingface.co/joshcx/static-embedding-all-mpnet-base-v2) | 256 |

Other static embedding models may work through the HuggingFace loader, but only the above models have been tested.