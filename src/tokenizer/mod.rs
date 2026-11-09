pub mod mask;

pub use mask::Mask;

#[derive(Clone)]
pub struct Token {
    pub content: String,
    pub mask: Mask,
}

impl Token {
    pub fn push(&mut self, c: char) {
        self.content.push(c);
    }
    pub fn space() -> Self {
        Self {
            content: " ".to_string(),
            mask: Mask::Whitespace,
        }
    }
}

impl Default for Token {
    fn default() -> Self {
        Self {
            content: String::new(),
            mask: Mask::None,
        }
    }
}

#[derive(Clone)]
pub struct TokenizedString {
    tokens: Vec<Token>,
}
impl TokenizedString {
    pub fn tokens(&self) -> Vec<Token> {
        self.tokens.clone()
    }
    pub fn push(&mut self, value: Token) {
        self.tokens.push(value);
    }
    pub fn text(&self) -> String {
        let mut out = String::new();

        for (i, token) in self.tokens.iter().enumerate() {
            out.push_str(&token.content);

            let next_is_word = self
                .tokens
                .get(i + 1)
                .is_some_and(|next| next.mask == Mask::None);

            if (token.mask == Mask::None && next_is_word) || token.mask == Mask::Punctuation {
                out.push(' ');
            }
        }

        out
    }
}

pub fn tokenize(input: &str) -> TokenizedString {
    let mut tokens: Vec<Token> = Vec::new();
    let mut cur_token = Token::default();
    let mut mask = Mask::Unknown;

    for c in input.chars() {
        let new_mask = Mask::classify_char(c);

        if new_mask != mask {
            flush(&mut tokens, &mut cur_token, mask);
            mask = new_mask.clone();
        }

        if new_mask != Mask::Whitespace {
            cur_token.push(c);
        }
    }

    flush(&mut tokens, &mut cur_token, mask);

    TokenizedString { tokens }
}

/// Pushes the current token (if any) onto `tokens` and resets it.
fn flush(tokens: &mut Vec<Token>, cur_token: &mut Token, mask: Mask) {
    if matches!(mask, Mask::Unknown | Mask::Whitespace) {
        *cur_token = Token::default();
        return;
    }
    let mut token = std::mem::take(cur_token);
    token.mask = mask;
    tokens.push(token);
}
