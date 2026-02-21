use std::sync::{Mutex, OnceLock};
use stembed::embedding::BERTLikeStaticEmbeddingModel;
use stembed::tokenizer::Vocab;
use wasm_bindgen::prelude::*;

extern crate console_error_panic_hook;
use std::panic;

static MODEL: OnceLock<Mutex<Option<BERTLikeStaticEmbeddingModel>>> = OnceLock::new();

/// Loads the model with the given vocabulary and model weights
///
/// # Arguments
/// * `vocab_data` - Newline delimited vocabulary string
/// * `model_data` - Raw bytes containing model weights
#[wasm_bindgen]
pub fn load(vocab_data: String, model_data: &[u8]) {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    let vocab = Vocab::from_newline_deliminated_string(&vocab_data);

    MODEL
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(PotionModel::potion_8m(
            vocab,
            model_data,
            stembed::embedding::embedder::Endianness::Little,
        ));
}

/// Embeds a single sentence into a vector of 32-bit floats
///
/// # Arguments
/// * `sentence` - Input text to embed
///
/// # Returns
/// Vector of f32 embedding values
#[wasm_bindgen]
pub fn embed_f32(sentence: String) -> Vec<f32> {
    let model = MODEL.get_or_init(|| Mutex::new(None)).lock().unwrap();
    model.as_ref().unwrap().embed_f32(&sentence)
}

/// Embeds a single sentence into a vector of 8-bit integers
///
/// # Arguments
/// * `sentence` - Input text to embed
///
/// # Returns
/// Vector of i8 embedding values
#[wasm_bindgen]
pub fn embed_i8(sentence: String) -> Vec<i8> {
    let model = MODEL.get_or_init(|| Mutex::new(None)).lock().unwrap();
    model.as_ref().unwrap().embed_i8(&sentence)
}

/// Embeds multiple sentences into vectors of 32-bit floats
///
/// # Arguments
/// * `sentences` - Vector of input texts to embed
///
/// # Returns
/// Flattened vector of f32 embedding values for all sentences
#[wasm_bindgen]
pub fn batch_embed_f32(sentences: Vec<String>) -> Vec<f32> {
    let model = MODEL.get_or_init(|| Mutex::new(None)).lock().unwrap();

    let mut embeddings = Vec::with_capacity(sentences.len() * model.as_ref().unwrap().dim);

    for sentence in sentences {
        embeddings.extend(model.as_ref().unwrap().embed_f32(&sentence));
    }

    embeddings
}

/// Embeds multiple sentences into vectors of 8-bit integers
///
/// # Arguments
/// * `sentences` - Vector of input texts to embed
///
/// # Returns
/// Flattened vector of i8 embedding values for all sentences
#[wasm_bindgen]
pub fn batch_embed_i8(sentences: Vec<String>) -> Vec<i8> {
    let model = MODEL.get_or_init(|| Mutex::new(None)).lock().unwrap();

    let mut embeddings = Vec::with_capacity(sentences.len() * model.as_ref().unwrap().dim);

    for sentence in sentences {
        embeddings.extend(model.as_ref().unwrap().embed_i8(&sentence));
    }

    embeddings
}
