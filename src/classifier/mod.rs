use crate::{
    Model, NUMERO_DE_POSICOES,
    tokenizer::{Mask, TokenizedString},
};
use hashbrown::HashMap;

// Changes inplace and returns anyway
pub fn classify(tokenized_string: &TokenizedString, model: &mut Model) {
    for token_window in tokenized_string.tokens().windows(NUMERO_DE_POSICOES + 1) {
        let root = token_window[0].clone();
        if root.mask != Mask::None {
            continue;
        }
        if !model.contains_key(&root.content) {
            model.insert(root.content.clone(), HashMap::new());
        }

        for (i, token) in token_window[1..].iter().enumerate() {
            let posicoes = model
                .entry(root.content.clone())
                .or_insert_with(HashMap::new);
            let relacoes_in_pos = posicoes.entry(i).or_insert_with(HashMap::new);

            *relacoes_in_pos
                .entry(token.content.clone())
                .or_insert_with(|| 0) += 1;
        }
    }
}
