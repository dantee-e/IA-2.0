use std::fmt::Display;

// copied from https://github.com/guillaume-be/rust-tokenizers
#[derive(Default, Clone, PartialEq)]
pub enum Mask {
    /// The token has no particular mask. This is the default situation. It may indicate that further processing can be done on a token.
    #[default]
    None,
    /// The token represents a whitespace (in any shape or form)
    Whitespace,
    /// The token represents punctuation (in any shape or form)
    Punctuation,
    /// The token is numeric
    Numeric,
    /// The token is out of vocabulary, it is unknown by the tokenizer and it will decode to unknown. Tokens that can be decoded properly (but may still be out of vocabulary) should not set this.
    Unknown,
}
impl Display for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Mask::None => String::from("None"),
            Mask::Whitespace => String::from("Whitespace"),
            Mask::Punctuation => String::from("Punctuation"),
            Mask::Unknown => String::from("Unknown"),
            Mask::Numeric => String::from("Numeric"),
        };

        write!(f, "{str}")
    }
}

impl Mask {
    pub fn classify_char(c: char) -> Self {
        if c.is_whitespace() {
            Mask::Whitespace
        } else if c.is_ascii_punctuation() {
            Mask::Punctuation
        } else if c.is_numeric() {
            Mask::Numeric
        } else {
            Mask::None
        }
    }

    pub fn classify_word(word: &str) -> Self {
        let c = word.chars().next().unwrap();

        if c.is_whitespace() {
            Mask::Whitespace
        } else if c.is_ascii_punctuation() {
            Mask::Punctuation
        } else if c.is_numeric() {
            Mask::Numeric
        } else {
            Mask::None
        }
    }
}
