use rand::seq::IndexedRandom;

use crate::{Mask, NUMERO_DE_POSICOES, Token, TrainingModel, tokenizer::TokenizedString};

// mutates inplace and returns the generated token
pub fn generate(tokens: &mut TokenizedString, model: &mut TrainingModel) -> Token {
    let mut candidatos: Vec<(&str, u32)> = Vec::new();
    for (i, token) in tokens
        .tokens()
        .iter()
        .rev()
        .take(NUMERO_DE_POSICOES)
        .enumerate()
    {
        candidatos.extend(model.followers(&token.content, i as u16));
    }

    let word = candidatos
        .choose_weighted(&mut rand::rng(), |c| c.1)
        .map_or("NO_WORD_FOUND", |c| c.0)
        .to_string();

    let token = Token {
        content: word.to_string(),
        mask: Mask::classify_word(&word),
    };

    tokens.push(token.clone());

    if token.mask == Mask::Punctuation {
        tokens.push(Token::space());
    }
    token
}
