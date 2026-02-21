use half::f16;

pub struct MeanStaticEmbedder {
    pub dim: usize,
    data: Box<[f32]>,
}

pub enum Endianness {
    Little,
    Big,
}

pub enum WeightsDType {
    F32,
    F16,
}

impl MeanStaticEmbedder {
    pub fn new(dim: usize, data: Box<[f32]>) -> Self {
        Self { dim, data }
    }

    pub fn get(&self, index: usize) -> &[f32] {
        &self.data[index * self.dim..(index + 1) * self.dim]
    }

    pub fn embed_f32(&self, tokens: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0; self.dim];

        for token in tokens {
            let embedding = self.get(*token);

            for i in 0..self.dim {
                out[i] += embedding[i];
            }
        }

        normalize(&out)
    }

    /// Embedding using i8 to the values [-127, 127]
    pub fn embed_i8(&self, tokens: &[usize]) -> Vec<i8> {
        let max_dim_value = (1.0 / self.dim as f32).sqrt();

        let embeddings = self.embed_f32(tokens);

        let mut out = vec![0i8; self.dim];

        for i in 0..self.dim {
            let scaled = (embeddings[i] / max_dim_value * 127.0) as i8;
            out[i] = scaled;
        }

        out
    }

    pub fn from_buffer_dynamic(
        dim: usize,
        buffer: &[u8],
        dtype: WeightsDType,
        endianness: Endianness,
    ) -> Self {
        match dtype {
            WeightsDType::F32 => Self::from_buffer_f32(dim, buffer, endianness),
            WeightsDType::F16 => Self::from_buffer_f16(dim, buffer, endianness),
        }
    }

    pub fn from_buffer_f32(dim: usize, buffer: &[u8], endianness: Endianness) -> Self {
        let mut data = Vec::with_capacity(dim);
        for i in (0..buffer.len()).step_by(4) {
            let bytes = buffer[i..i + 4].try_into().unwrap();
            let value = match endianness {
                Endianness::Little => f32::from_le_bytes(bytes),
                Endianness::Big => f32::from_be_bytes(bytes),
            };
            data.push(value);
        }

        Self::new(dim, data.into_boxed_slice())
    }

    pub fn from_buffer_f16(dim: usize, buffer: &[u8], endianness: Endianness) -> Self {
        let mut data = Vec::with_capacity(dim);
        for i in (0..buffer.len()).step_by(2) {
            let bytes = buffer[i..i + 2].try_into().unwrap();
            let value = match endianness {
                Endianness::Little => f16::from_le_bytes(bytes),
                Endianness::Big => f16::from_be_bytes(bytes),
            };
            data.push(value.to_f32());
        }

        Self::new(dim, data.into_boxed_slice())
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-8;
    v.iter().map(|x| x / norm).collect()
}
