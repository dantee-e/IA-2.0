use std::time::{Duration, Instant};

use rand::seq::IndexedRandom;

use crate::{Mask, NUMBER_OF_POSITIONS, Token, model::Model, tokenizer::TokenizedString};

#[derive(Default)]
pub struct GeneratorTimer {
    pub accessing_storage: Duration,
    pub choosing_candidate: Duration,
}

// mutates inplace and returns the generated token
// TODO fazer o peso ser maior quanto mais perto a palavra
pub fn generate_with_timer(
    tokens: &mut TokenizedString,
    model: &Model,
    timer: &mut GeneratorTimer,
) -> Token {
    let mut candidatos: Vec<(&str, f32)> = Vec::new();

    let mut start = Instant::now();
    for (i, token) in tokens
        .tokens()
        .iter()
        .rev()
        .take(NUMBER_OF_POSITIONS)
        .enumerate()
    {
        candidatos.extend(model.followers(&token.content, i as u16));
    }
    timer.accessing_storage += start.elapsed();

    start = Instant::now();

    let word = candidatos
        .choose_weighted(&mut rand::rng(), |c| c.1)
        .map_or("NO_WORD_FOUND", |c| c.0)
        .to_string();

    timer.choosing_candidate += start.elapsed();

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

// mutates inplace and returns the generated token
// TODO fazer o peso ser maior quanto mais perto a palavra
pub fn generate(tokens: &mut TokenizedString, model: &Model) -> Token {
    let mut candidatos: Vec<(&str, f32)> = Vec::new();

    for (i, token) in tokens
        .tokens()
        .iter()
        .rev()
        .take(NUMBER_OF_POSITIONS)
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
