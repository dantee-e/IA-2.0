use crate::{
    NUMERO_DE_POSICOES, TrainingModel,
    tokenizer::{Mask, TokenizedString},
};

// Changes inplace
pub fn classify(tokenized_string: &TokenizedString, model: &mut TrainingModel) {
    for window in tokenized_string.tokens().windows(NUMERO_DE_POSICOES + 1) {
        let root = &window[0];
        if root.mask != Mask::None {
            continue;
        }

        for (i, token) in window[1..].iter().enumerate() {
            model.insert_or_count((root.content.clone(), i as u16), token.content.clone());
        }
    }
}
