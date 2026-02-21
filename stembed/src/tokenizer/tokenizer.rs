pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<usize>;
}
