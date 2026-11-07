use hashbrown::HashMap;
use rand::seq::IndexedRandom;

use crate::{Mask, NUMERO_DE_POSICOES, Token, TrainingModel, tokenizer::TokenizedString};

// mutates inplace and returns the generated token
pub fn generate(tokens: &mut TokenizedString, model: &mut TrainingModel) -> Token {
    let last_n = tokens.tokens().into_iter().rev().take(NUMERO_DE_POSICOES);

    let mut lista_candidatos: HashMap<String, usize> = HashMap::new();

    for (i, token) in last_n.enumerate() {
        let id = if let Some(id) = model.id_from_key_option(&(token.content.clone(), i as u16)) {
            id
        } else {
            continue;
        };
        if let Some(relations) = model.get_from_id(id) {
            for candidate in relations {
                *lista_candidatos
                    .entry(candidate.0.clone())
                    .or_insert(candidate.1) += 1
            }
        }
    }

    let mut rng = rand::rng();

    let candidatos = Vec::from_iter(lista_candidatos);

    let word = candidatos
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
