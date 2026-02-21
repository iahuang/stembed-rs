use std::collections::HashMap;
use crate::tokenizer::Vocab;
use super::{ModelBinaryError, StaticEmbeddingModelBuilder, WeightsDType};

impl StaticEmbeddingModelBuilder {
    /// Serializes the model to a binary format
    /// [dim: u32, dtype: u8, unk_token_id: u32, vocab_size: u32, vocab: { bytelen: u32, content: u8[bytelen] }[vocab_size], weights: f16
    /// Where dtype is
    /// - 0 for f32
    /// - 1 for f16
    pub fn to_model_binary(&self) -> Box<[u8]> {
        let mut data = Vec::new();
        let dim_u32: u32 = self
            .dim
            .try_into()
            .expect("model dim must fit into u32 for binary serialization");
        let unk_u32: u32 = self
            .vocab
            .unk_token_id
            .try_into()
            .expect("unk_token_id must fit into u32 for binary serialization");
        let vocab_size = self.vocab.vocab.len();
        let vocab_size_u32: u32 = vocab_size
            .try_into()
            .expect("vocab size must fit into u32 for binary serialization");
        let dtype_u8 = match self.weights_dtype {
            WeightsDType::F32 => 0,
            WeightsDType::F16 => 1,
        };

        let bytes_per_weight = match self.weights_dtype {
            WeightsDType::F32 => 4,
            WeightsDType::F16 => 2,
        };
        let expected_weights_len = self
            .dim
            .checked_mul(vocab_size)
            .and_then(|n| n.checked_mul(bytes_per_weight))
            .expect("dim * vocab_size * dtype_size overflow during serialization");
        assert_eq!(
            self.weights.len(),
            expected_weights_len,
            "weights length does not match dim * vocab_size * dtype_size"
        );

        let mut tokens_by_id: Vec<Option<&str>> = vec![None; vocab_size];
        for (token, id) in &self.vocab.vocab {
            assert!(
                *id < vocab_size,
                "vocab id {id} is out of range for vocab size {vocab_size}"
            );
            assert!(
                tokens_by_id[*id].is_none(),
                "duplicate vocab id {id} in vocabulary map"
            );
            tokens_by_id[*id] = Some(token.as_str());
        }

        data.extend_from_slice(&dim_u32.to_le_bytes());
        data.push(dtype_u8);
        data.extend_from_slice(&unk_u32.to_le_bytes());
        data.extend_from_slice(&vocab_size_u32.to_le_bytes());

        for token in tokens_by_id {
            let token = token.expect("vocab ids must be contiguous from 0..vocab_size");
            let token_len_u32: u32 = token
                .len()
                .try_into()
                .expect("token byte length must fit into u32");
            data.extend_from_slice(&token_len_u32.to_le_bytes());
            data.extend_from_slice(token.as_bytes());
        }

        data.extend_from_slice(&self.weights);
        data.into_boxed_slice()
    }

    pub fn from_model_binary(data: &[u8]) -> Result<Self, ModelBinaryError> {
        let mut offset = 0usize;

        fn safe_read_u8(data: &[u8], offset: &mut usize) -> Result<u8, ModelBinaryError> {
            if *offset + 1 > data.len() {
                return Err(ModelBinaryError::UnexpectedEof);
            }
            let value = data[*offset];
            *offset += 1;
            Ok(value)
        }

        fn safe_read_u32(data: &[u8], offset: &mut usize) -> Result<u32, ModelBinaryError> {
            if *offset + 4 > data.len() {
                return Err(ModelBinaryError::UnexpectedEof);
            }
            let bytes = data[*offset..*offset + 4]
                .try_into()
                .map_err(|_| ModelBinaryError::UnexpectedEof)?;
            *offset += 4;
            Ok(u32::from_le_bytes(bytes))
        }

        fn safe_read_bytes<'a>(
            data: &'a [u8],
            offset: &mut usize,
            len: usize,
        ) -> Result<&'a [u8], ModelBinaryError> {
            if *offset + len > data.len() {
                return Err(ModelBinaryError::UnexpectedEof);
            }
            let out = &data[*offset..*offset + len];
            *offset += len;
            Ok(out)
        }

        let dim = safe_read_u32(data, &mut offset)? as usize;
        let weights_dtype = match safe_read_u8(data, &mut offset)? {
            0 => WeightsDType::F32,
            1 => WeightsDType::F16,
            other => return Err(ModelBinaryError::InvalidDType(other)),
        };
        let unk_token_id = safe_read_u32(data, &mut offset)? as usize;
        let vocab_size = safe_read_u32(data, &mut offset)? as usize;

        let mut vocab = HashMap::with_capacity(vocab_size);
        for i in 0..vocab_size {
            let token_len = safe_read_u32(data, &mut offset)? as usize;
            let token_bytes = safe_read_bytes(data, &mut offset, token_len)?;
            let token = std::str::from_utf8(token_bytes).map_err(ModelBinaryError::InvalidUtf8)?;
            vocab.insert(token.to_owned(), i);
        }

        let weights = data[offset..].to_vec().into_boxed_slice();
        let bytes_per_weight = match weights_dtype {
            WeightsDType::F32 => 4,
            WeightsDType::F16 => 2,
        };
        let expected_weights_len = dim
            .checked_mul(vocab_size)
            .and_then(|n| n.checked_mul(bytes_per_weight))
            .ok_or(ModelBinaryError::Overflow)?;

        if weights.len() != expected_weights_len {
            return Err(ModelBinaryError::InvalidWeightsLength {
                expected: expected_weights_len,
                actual: weights.len(),
            });
        }

        Ok(Self {
            dim,
            vocab: Vocab::new(vocab, unk_token_id),
            weights,
            weights_dtype,
        })
    }
}
