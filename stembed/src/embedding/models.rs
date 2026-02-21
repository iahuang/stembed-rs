use super::{
    base_model::BaseStaticEmbeddingModel,
    embedder::{Endianness, MeanStaticEmbedder, WeightsDType},
};

use crate::tokenizer::{Vocab, WordPieceTokenizer, normalizer::BertNormalizer};
use half::f16;
use std::collections::HashMap;

/// Any static embedding model which uses the BERT uncased tokenizer and mean pooling.
/// From what I can tell, this is the case for the vast majority of static embedding models.
pub type BERTLikeStaticEmbeddingModel =
    BaseStaticEmbeddingModel<WordPieceTokenizer<BertNormalizer>>;

#[derive(Debug)]
pub struct StaticEmbeddingModelBuilder {
    pub dim: usize,
    pub vocab: Vocab,
    pub weights: Box<[u8]>,
    pub weights_dtype: WeightsDType,
}

#[derive(Debug)]
pub enum ModelBinaryError {
    UnexpectedEof,
    InvalidDType(u8),
    InvalidUtf8(std::str::Utf8Error),
    InvalidWeightsLength { expected: usize, actual: usize },
    Overflow,
}

impl StaticEmbeddingModelBuilder {
    pub fn new(dim: usize, vocab: Vocab, weights: Box<[u8]>, weights_dtype: WeightsDType) -> Self {
        Self {
            dim,
            vocab,
            weights,
            weights_dtype,
        }
    }

    pub fn cast_weights_dtype(&self, weights_dtype: WeightsDType) -> Self {
        let weights = match (&self.weights_dtype, &weights_dtype) {
            (WeightsDType::F32, WeightsDType::F32) | (WeightsDType::F16, WeightsDType::F16) => {
                self.weights.clone()
            }
            (WeightsDType::F32, WeightsDType::F16) => {
                assert_eq!(
                    self.weights.len() % 4,
                    0,
                    "f32 weights buffer length must be divisible by 4"
                );

                let mut out = Vec::with_capacity(self.weights.len() / 2);
                for chunk in self.weights.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into().expect("chunk size is always 4");
                    let value = f32::from_le_bytes(bytes);
                    out.extend_from_slice(&f16::from_f32(value).to_le_bytes());
                }
                out.into_boxed_slice()
            }
            (WeightsDType::F16, WeightsDType::F32) => {
                assert_eq!(
                    self.weights.len() % 2,
                    0,
                    "f16 weights buffer length must be divisible by 2"
                );

                let mut out = Vec::with_capacity(self.weights.len() * 2);
                for chunk in self.weights.chunks_exact(2) {
                    let bytes: [u8; 2] = chunk.try_into().expect("chunk size is always 2");
                    let value = f16::from_le_bytes(bytes).to_f32();
                    out.extend_from_slice(&value.to_le_bytes());
                }
                out.into_boxed_slice()
            }
        };

        Self {
            dim: self.dim,
            vocab: Vocab::new(self.vocab.vocab.clone(), self.vocab.unk_token_id),
            weights,
            weights_dtype,
        }
    }
}

pub fn base_model_with_bert_uncased_tokenizer(
    builder: StaticEmbeddingModelBuilder,
) -> BERTLikeStaticEmbeddingModel {
    let unk_token_id = builder.vocab.unk_token_id;
    let tokenizer = WordPieceTokenizer::new(
        builder.vocab,
        unk_token_id,
        BertNormalizer::new(true, true, None, true),
    );
    let embedder = MeanStaticEmbedder::from_buffer_dynamic(
        builder.dim,
        &builder.weights,
        builder.weights_dtype,
        Endianness::Little,
    );
    BaseStaticEmbeddingModel::new(builder.dim, embedder, tokenizer)
}

/// A pretrained Potion-Base-8M model instance. Requires a feature flag to be enabled
/// to bundle the model weights into the binary.
#[cfg(feature = "potion-base-8m")]
pub fn pretrained_potion_base_8m() -> BERTLikeStaticEmbeddingModel {
    let vocab = Vocab::from_newline_deliminated_string(
        include_str!("../../../include/vocab_potion.txt"),
        1,
    );
    let weights = include_bytes!("../../../include/potion_base_8M.f16.bin");

    base_model_with_bert_uncased_tokenizer(StaticEmbeddingModelBuilder {
        dim: 256,
        vocab,
        weights: weights.to_vec().into_boxed_slice(),
        weights_dtype: WeightsDType::F16,
    })
}
