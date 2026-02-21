use super::normalizer::Normalizer;
use super::tokenizer::Tokenizer;
use super::vocab::Vocab;

pub struct WordPieceTokenizer<N: Normalizer> {
    vocab: Vocab,
    unk_token_id: usize,
    normalizer: N,
}

impl<N: Normalizer> WordPieceTokenizer<N> {
    pub fn new(vocab: Vocab, unk_token_id: usize, normalizer: N) -> Self {
        Self {
            vocab,
            unk_token_id,
            normalizer,
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let normalized_text = self.normalizer.normalize(text);
        let mut output = Vec::<usize>::new();

        for word in normalized_text.split(|c| self.normalizer.split_fn(c)) {
            let word_chars = word.chars().collect::<Vec<_>>();
            let mut start = 0;
            let mut sub_tokens = Vec::<usize>::new();

            while start < word_chars.len() {
                let mut end = word_chars.len();
                let mut matched_token = None;

                while start < end {
                    let substr = {
                        if start > 0 {
                            &format!(
                                "##{}",
                                &word_chars
                                    .iter()
                                    .skip(start)
                                    .take(end - start)
                                    .collect::<String>()
                            )
                        } else {
                            &word_chars
                                .iter()
                                .skip(start)
                                .take(end - start)
                                .collect::<String>()
                        }
                    };

                    if let Some(token_id) = self.vocab.get_token_to_id(substr) {
                        matched_token = Some(token_id);
                        break;
                    }

                    end -= 1;
                }

                if let Some(matched_token) = matched_token {
                    sub_tokens.push(matched_token);
                    start = end;
                } else {
                    sub_tokens.push(self.unk_token_id);
                    break;
                }
            }

            output.extend(sub_tokens.iter());
        }
        output
    }
}

impl<N: Normalizer> Tokenizer for WordPieceTokenizer<N> {
    fn tokenize(&self, text: &str) -> Vec<usize> {
        self.tokenize(text)
    }
}
