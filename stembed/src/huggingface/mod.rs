use hf_hub::api::sync::{Api, ApiError};

#[derive(Debug)]
pub enum HuggingFaceModelError {
    UnsupportedModel(String),
    IOError(std::io::Error),
    HuggingFaceApiError(ApiError),
    ParseError(serde_json::Error),
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

#[derive(Debug, serde::Deserialize)]
struct HFModule {
    r#type: String,
    path: String,
}

pub fn hf_model_as_builder_binary(model_id: &str) -> Result<Box<[u8]>, HuggingFaceModelError> {
    let api = Api::new()?;
    let repo = api.model(model_id.to_string());

    let modules = repo.get("modules.json")?;
    let modules_file = std::fs::File::open(modules)?;
    let modules: Vec<HFModule> = serde_json::from_reader(modules_file)?;

    let sentence_transformers = modules
        .iter()
        .filter(|m| m.r#type == "sentence_transformer.SentenceTransformer")
        .collect::<Vec<_>>();

    if sentence_transformers.len() != 1 {
        return Err(HuggingFaceModelError::UnsupportedModel(format!(
            "Expected exactly one SentenceTransformer module, got {}",
            sentence_transformers.len()
        )));
    }

    let sentence_transformer = sentence_transformers[0];
    
}
