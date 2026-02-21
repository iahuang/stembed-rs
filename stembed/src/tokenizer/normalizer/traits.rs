pub trait Normalizer {
    fn normalize(&self, text: &str) -> String;
    fn split_fn(&self, c: char) -> bool;
}
