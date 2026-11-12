use crate::{
    NUMBER_OF_POSITIONS, TrainingModel,
    tokenizer::{Mask, TokenizedString},
};

fn windows_to_end<T>(slice: &[T], n: usize) -> impl Iterator<Item = &[T]> {
    (0..slice.len()).map(move |i| &slice[i..(i + n).min(slice.len())])
}

// Changes inplace
pub fn classify(tokenized_string: &TokenizedString, model: &mut TrainingModel) {
    for window in windows_to_end(tokenized_string.tokens(), NUMBER_OF_POSITIONS + 1) {
        let root = &window[0];
        if root.mask != Mask::None {
            continue;
        }

        for (i, token) in window[1..].iter().enumerate() {
            model.insert_or_increase_count(&root.content, i as u16, &token.content)
        }
    }
}
