use std::time::{Duration, Instant};

use hashbrown::HashMap;
use rand::seq::IndexedRandom;

use crate::{Mask, NUMBER_OF_POSITIONS, Token, model::Model, tokenizer::TokenizedString};

#[derive(Default)]
pub struct GeneratorTimer {
    pub accessing_storage: Duration,
    pub choosing_candidate: Duration,
}

// mutates inplace and returns the generated token
pub fn generate_with_timer(
    tokens: &mut TokenizedString,
    model: &Model,
    timer: &mut GeneratorTimer,
) -> Token {
    // Merge candidates across all context positions: a word seen at several
    // distances gets the sum of its position-weighted counts.
    let mut scores: HashMap<&str, f32> = HashMap::new();

    let mut start = Instant::now();
    for (i, token) in tokens
        .tokens()
        .iter()
        .rev()
        .take(NUMBER_OF_POSITIONS)
        .enumerate()
    {
        for (word, weight) in model.followers(&token.content, i as u16) {
            *scores.entry(word).or_insert(0.0) += weight;
        }
    }
    timer.accessing_storage += start.elapsed();

    start = Instant::now();

    let candidatos: Vec<(&str, f32)> = scores.into_iter().collect();
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
pub fn generate(tokens: &mut TokenizedString, model: &Model) -> Token {
    generate_with_timer(tokens, model, &mut GeneratorTimer::default())
}
