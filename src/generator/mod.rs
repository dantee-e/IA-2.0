use rand::seq::IndexedRandom;

use crate::{Mask, NUMERO_DE_POSICOES, Token, TrainingModel, tokenizer::TokenizedString};

// mutates inplace and returns the generated token
pub fn generate(tokens: &mut TokenizedString, model: &TrainingModel) -> Token {
    let last_n = tokens.tokens().into_iter().rev().take(NUMERO_DE_POSICOES);

    let mut lista_candidatos: Vec<&(String, usize)> = Vec::new();

    for (i, token) in last_n.enumerate() {
        if let Some(relations) = model.get((token.content, i)) {
            for candidate in relations {
                lista_candidatos.push(candidate);
            }
        }
    }

    let mut rng = rand::rng();

    let word = lista_candidatos
        .choose_weighted(&mut rng, |c| c.1)
        .map_or("NO_WORD_FOUND".to_string(), |c| c.0.clone());

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
