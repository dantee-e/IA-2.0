use crate::{
    NUMBER_OF_POSITIONS, TrainingModel,
    model::{Bucket, increase_count},
    tokenizer::{Mask, TokenizedString},
};
use hashbrown::HashMap;

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

/// Parallel variant of `classify`: records only the windows whose root word id
/// satisfies `root_id % num_shards == shard_id`, into that shard's own bucket
/// array (indexed by `(root_id / num_shards) * NUMBER_OF_POSITIONS + pos`).
/// `vocab` must already contain every token of the corpus (see `build_vocab`),
/// so this performs lookups only and needs no synchronization across shards.
pub fn classify_shard(
    tokenized_string: &TokenizedString,
    vocab: &HashMap<String, u32>,
    shard_id: usize,
    num_shards: usize,
    shard: &mut [Bucket],
) {
    for window in windows_to_end(tokenized_string.tokens(), NUMBER_OF_POSITIONS + 1) {
        let root = &window[0];
        if root.mask != Mask::None {
            continue;
        }
        let Some(&root_id) = vocab.get(&root.content) else {
            continue;
        };
        if root_id as usize % num_shards != shard_id {
            continue;
        }
        let base = (root_id as usize / num_shards) * NUMBER_OF_POSITIONS;

        for (i, token) in window[1..].iter().enumerate() {
            let Some(&follower_id) = vocab.get(token.content.as_str()) else {
                continue;
            };
            increase_count(&mut shard[base + i], follower_id);
        }
    }
}
