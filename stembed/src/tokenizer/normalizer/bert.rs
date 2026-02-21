/**
 * Adapted from https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/normalizers/bert.rs
 */
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

use super::traits::Normalizer;

/// Checks whether a character is whitespace
fn is_whitespace(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    match c {
        '\t' | '\n' | '\r' => true,
        _ => c.is_whitespace(),
    }
}

/// Checks whether a character is a control character
fn is_control(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    match c {
        '\t' | '\n' | '\r' => false,
        // The definition of `is_control` here is quite large and contains also
        // Cc, Cf, Cn or Co
        // cf. https://unicode.org/reports/tr44/ (Table 12)
        _ => c.is_other(),
    }
}

/// Checks whether a character is chinese
/// This defines a "chinese character" as anything in the CJK Unicode block:
///   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
///
/// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
/// despite its name. The modern Korean Hangul alphabet is a different block,
/// as is Japanese Hiragana and Katakana. Those alphabets are used to write
/// space-separated words, so they are not treated specially and handled
/// like for all of the other languages.
fn is_chinese_char(c: char) -> bool {
    matches!(
        c as usize,
        0x4E00..=0x9FFF |
        0x3400..=0x4DBF |
        0x20000..=0x2A6DF |
        0x2A700..=0x2B73F |
        0x2B740..=0x2B81F |
        0x2B920..=0x2CEAF |
        0xF900..=0xFAFF |
        0x2F800..=0x2FA1F
    )
}

#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct BertNormalizer {
    /// Whether to do the bert basic cleaning:
    ///   1. Remove any control characters
    ///   2. Replace all sorts of whitespace by the classic one ` `
    pub clean_text: bool,
    /// Whether to put spaces around chinese characters so they get split
    pub handle_chinese_chars: bool,
    /// Whether to strip accents
    pub strip_accents: Option<bool>,
    /// Whether to lowercase the input
    pub lowercase: bool,
}

impl BertNormalizer {
    pub fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> Self {
        Self {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        }
    }

    fn do_clean_text(&self, normalized: &mut String) -> String {
        normalized.retain(|c| !(c as usize == 0 || c as usize == 0xfffd || is_control(c)));
        normalized
            .chars()
            .map(|c| if is_whitespace(c) { ' ' } else { c })
            .collect::<String>()
    }

    fn do_handle_chinese_chars(&self, normalized: &mut String) -> String {
        let mut new_string = String::new();
        normalized.chars().for_each(|c| {
            if is_chinese_char(c) {
                new_string.push_str(" ");
                new_string.push(c);
                new_string.push_str(" ");
            } else {
                new_string.push(c);
            }
        });
        new_string
    }

    fn do_handle_bert_punct(&self, normalized: &mut String) -> String {
        let mut new_string = String::new();
        normalized.chars().for_each(|c| {
            if c.is_punctuation() || c.is_ascii_whitespace() {
                new_string.push_str(" ");
                new_string.push(c);
                new_string.push_str(" ");
            } else {
                new_string.push(c);
            }
        });
        new_string
    }

    fn do_strip_accents(&self, normalized: &mut String) -> String {
        normalized
            .nfd()
            .map(|c| c.0)
            .filter(|c| !c.is_mark_nonspacing())
            .collect::<String>()
    }

    fn do_lowercase(&self, normalized: &mut String) -> String {
        normalized.to_lowercase()
    }
}

impl Normalizer for BertNormalizer {
    fn normalize(&self, text: &str) -> String {
        let mut normalized = text.to_string();
        if self.clean_text {
            normalized = self.do_clean_text(&mut normalized);
        }
        if self.handle_chinese_chars {
            normalized = self.do_handle_chinese_chars(&mut normalized);
        }
        if self.strip_accents.unwrap_or(self.lowercase) {
            normalized = self.do_strip_accents(&mut normalized);
        }
        if self.lowercase {
            normalized = self.do_lowercase(&mut normalized);
        }

        normalized = self.do_handle_bert_punct(&mut normalized);
        normalized
    }

    fn split_fn(&self, c: char) -> bool {
        c.is_whitespace()
    }
}
