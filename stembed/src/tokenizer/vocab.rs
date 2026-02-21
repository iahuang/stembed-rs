use std::collections::HashMap;

pub struct Vocab {
    pub vocab: HashMap<String, usize>,
    pub unk_token_id: usize,
}

impl Vocab {
    pub fn new(vocab: HashMap<String, usize>, unk_token_id: usize) -> Self {
        Self { vocab, unk_token_id }
    }

    pub fn from_newline_deliminated_string(data: &str, unk_token_id: usize) -> Self {
        let vocab: HashMap<String, usize> = data
            .split("\n")
            .filter(|line| !line.is_empty())
            .enumerate()
            .map(|(i, line)| (line.to_string(), i))
            .collect();
        Self::new(vocab, unk_token_id)
    }

    pub fn get_token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).cloned()
    }

    pub fn get_id_to_token(&self, id: usize) -> Option<&String> {
        self.vocab.iter().find(|(_, v)| **v == id).map(|(k, _)| k)
    }
}
