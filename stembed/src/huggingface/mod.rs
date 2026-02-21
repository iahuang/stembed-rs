use std::collections::HashMap;

use crate::{
    embedding::{StaticEmbeddingModelBuilder, WeightsDType},
    tokenizer::Vocab,
};
use hf_hub::api::sync::{Api, ApiError};

#[derive(Debug)]
pub enum HuggingFaceModelError {
    UnsupportedModel(String),
    IOError(std::io::Error),
    HuggingFaceApiError(ApiError),
    ParseError(serde_json::Error),
    SafetensorsError(safetensors::SafeTensorError),
}

impl From<std::io::Error> for HuggingFaceModelError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}

impl From<ApiError> for HuggingFaceModelError {
    fn from(value: ApiError) -> Self {
        Self::HuggingFaceApiError(value)
    }
}

impl From<serde_json::Error> for HuggingFaceModelError {
    fn from(value: serde_json::Error) -> Self {
        Self::ParseError(value)
    }
}

impl From<safetensors::SafeTensorError> for HuggingFaceModelError {
    fn from(value: safetensors::SafeTensorError) -> Self {
        Self::SafetensorsError(value)
    }
}

#[derive(Debug, serde::Deserialize)]
struct HFModule {
    r#type: String,
    path: String,
}

#[derive(Debug, serde::Deserialize)]
struct HFTokenizerModel {
    vocab: HashMap<String, usize>,
    unk_token: String,
}

#[derive(Debug, serde::Deserialize)]
struct HFTokenizer {
    model: HFTokenizerModel,
}

fn hf_path_join(a: &str, b: &str) -> String {
    if a == "." {
        return b.to_string();
    }

    if a.ends_with('/') {
        a.to_string() + b
    } else {
        a.to_string() + "/" + b
    }
}

pub fn hf_model_as_builder(
    model_id: &str,
) -> Result<StaticEmbeddingModelBuilder, HuggingFaceModelError> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let modules_path = repo.get("modules.json")?;
    let modules_file = std::fs::File::open(modules_path)?;
    let modules: Vec<HFModule> = serde_json::from_reader(modules_file)?;

    let sentence_transformers = modules
        .iter()
        .filter(|m| m.r#type == "sentence_transformers.models.StaticEmbedding")
        .collect::<Vec<_>>();

    if sentence_transformers.len() != 1 {
        return Err(HuggingFaceModelError::UnsupportedModel(format!(
            "Expected exactly one SentenceTransformer module, got {}",
            sentence_transformers.len()
        )));
    }

    let sentence_transformer = sentence_transformers[0];
    let safetensors_path =
        repo.get(hf_path_join(&sentence_transformer.path, "model.safetensors").as_str())?;
    let safetensors_buffer = std::fs::read(safetensors_path)?;
    let safetensors = safetensors::SafeTensors::deserialize(safetensors_buffer.as_slice())?;
    let embedding_weights = safetensors
        .tensor("embedding.weight")
        .or(safetensors.tensor("embeddings"))?;
    let embedding_weights_shape = embedding_weights.shape();

    if embedding_weights_shape.len() != 2 {
        return Err(HuggingFaceModelError::UnsupportedModel(format!(
            "Expected embedding weights to have 2 dimensions, got {}",
            embedding_weights_shape.len()
        )));
    }

    let dim = embedding_weights_shape[1];
    let weights_dtype = match embedding_weights.dtype() {
        safetensors::Dtype::F16 => WeightsDType::F16,
        safetensors::Dtype::F32 => WeightsDType::F32,
        _ => {
            return Err(HuggingFaceModelError::UnsupportedModel(format!(
                "Expected embedding weights to be f16 or f32, got {}",
                embedding_weights.dtype().to_string()
            )));
        }
    };

    let tokenizer_path =
        repo.get(hf_path_join(&sentence_transformer.path, "tokenizer.json").as_str())?;
    let tokenizer_file = std::fs::File::open(tokenizer_path)?;
    let tokenizer: HFTokenizer = serde_json::from_reader(tokenizer_file)?;
    let unk_token_id = *tokenizer
        .model
        .vocab
        .get(&tokenizer.model.unk_token)
        .ok_or(HuggingFaceModelError::UnsupportedModel(format!(
            "Could not find unk_token in tokenizer: {}",
            tokenizer.model.unk_token
        )))?;

    Ok(StaticEmbeddingModelBuilder {
        dim,
        weights: embedding_weights.data().to_vec().into_boxed_slice(),
        weights_dtype,
        vocab: Vocab::new(tokenizer.model.vocab, unk_token_id),
    })
}
