use hashbrown::HashMap;
use rand::seq::IndexedRandom;

use crate::{Mask, Model, NUMERO_DE_POSICOES, Token, tokenizer::TokenizedString};

// mutates inplace and returns the generated token
pub fn generate(tokens: &mut TokenizedString, model: &Model) -> Token {
    let last_n = tokens.tokens().into_iter().rev().take(NUMERO_DE_POSICOES);

    let mut lista_candidatos: HashMap<String, usize> = HashMap::new();

    for (i, token) in last_n.enumerate() {
        if let Some(relations) = model
            .get(&token.content)
            .and_then(|positions| positions.get(&i))
        {
            for (word, count) in relations {
                *lista_candidatos.entry(word.clone()).or_insert(0) += count;
            }
        }
    }

    let mut rng = rand::rng();

    let candidatos: Vec<(String, usize)> = lista_candidatos.into_iter().collect();

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
