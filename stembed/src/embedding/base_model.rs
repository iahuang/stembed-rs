use crate::tokenizer::tokenizer::Tokenizer;

use super::embedder::MeanStaticEmbedder;

pub struct BaseStaticEmbeddingModel<T: Tokenizer> {
    embedder: MeanStaticEmbedder,
    tokenizer: T,
    pub dim: usize,
}

impl<T: Tokenizer> BaseStaticEmbeddingModel<T> {
    pub fn new(dim: usize, embedder: MeanStaticEmbedder, tokenizer: T) -> Self {
        Self {
            dim,
            embedder,
            tokenizer,
        }
    }

    pub fn embed_f32(&self, sentence: &str) -> Vec<f32> {
        let tokens = self.tokenizer.tokenize(sentence);
        self.embedder.embed_f32(&tokens)
    }

    pub fn embed_i8(&self, sentence: &str) -> Vec<i8> {
        let tokens = self.tokenizer.tokenize(sentence);
        self.embedder.embed_i8(&tokens)
    }

    pub fn tokenize(&self, sentence: &str) -> Vec<usize> {
        self.tokenizer.tokenize(sentence)
    }
}
