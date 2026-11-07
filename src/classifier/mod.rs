use crate::{
    Model, NUMERO_DE_POSICOES,
    tokenizer::{Mask, TokenizedString},
};

// Changes inplace and returns anyway
pub fn classify(tokenized_string: &TokenizedString, model: &mut Model) {
    for window in tokenized_string.tokens().windows(NUMERO_DE_POSICOES + 1) {
        let root = &window[0];
        if root.mask != Mask::None {
            continue;
        }

        let posicoes = model.entry(root.content.clone()).or_default();
        for (i, token) in window[1..].iter().enumerate() {
            *posicoes
                .entry(i)
                .or_default()
                .entry(token.content.clone())
                .or_insert(0) += 1;
        }
    }
}
