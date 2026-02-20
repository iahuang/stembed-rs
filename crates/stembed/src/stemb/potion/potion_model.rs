use crate::stemb::embedding::{Endianness, MeanStaticEmbedder};
use crate::tokenizer::{Vocab, WordPieceTokenizer, normalizer::BertNormalizer};

pub struct PotionModel {
    tokenizer: WordPieceTokenizer<BertNormalizer>,
    embedding: MeanStaticEmbedder,
    pub dim: usize,
}

impl PotionModel {
    pub fn new(dim: usize, vocab: Vocab, data: &[u8], endianness: Endianness) -> Self {
        let tokenizer =
            WordPieceTokenizer::new(vocab, 1, BertNormalizer::new(true, true, None, true));
        let embedding = MeanStaticEmbedder::from_buffer(dim, data, endianness);

        assert_eq!(embedding.dim, dim);

        Self {
            tokenizer,
            embedding,
            dim,
        }
    }

    pub fn embed_f32(&self, sentence: &str) -> Vec<f32> {
        let tokens = self.tokenizer.tokenize(sentence);
        println!("Tokens: {:?}", tokens);
        self.embedding.embed_f32(&tokens)
    }

    pub fn embed_quant_i8(&self, sentence: &str) -> Vec<i8> {
        let tokens = self.tokenizer.tokenize(sentence);
        self.embedding.embed_quant_i8(&tokens)
    }

    /// Constructor for Potion-Base-8M
    pub fn potion_8m(vocab: Vocab, data: &[u8], endianness: Endianness) -> PotionModel {
        PotionModel::new(256, vocab, data, endianness)
    }

    #[cfg(feature = "potion-base-8m")]
    pub fn pretrained_potion_base_8m() -> PotionModel {
        PotionModel::potion_8m(
            Vocab::from_newline_deliminated_string(include_str!("../../../../data/vocab.txt")),
            include_bytes!("../../../../data/potion_base_8M.bin"),
            crate::stemb::embedding::Endianness::Little,
        )
    }
}
