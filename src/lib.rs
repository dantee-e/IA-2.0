use std::io;
use std::path::{Path, PathBuf};

mod classifier;
pub mod generator;
mod model;
mod tokenizer;

pub use classifier::{classify, classify_shard};
pub use generator::generate;
pub use model::persist::{
    ParallelCheckpoint, load_parallel_checkpoint, load_training_model, save_parallel_checkpoint,
    save_training_model,
};
pub use model::{Bucket, Model, TrainingModel};
pub use tokenizer::Mask;
pub use tokenizer::Token;
pub use tokenizer::tokenize;

use hashbrown::HashMap;
use rayon::prelude::*;

pub const NUMBER_OF_POSITIONS: usize = 300;

fn list_files(dir: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort(); // deterministic vocab ids and bucket contents across runs
    Ok(files)
}

pub fn train_from_dir(dir: impl AsRef<Path>) -> io::Result<TrainingModel> {
    let mut model: TrainingModel = TrainingModel::default();

    for path in list_files(dir)? {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("skipping {}: {e}", path.display());
                continue;
            }
        };

        let tokenized = tokenize(&text);
        classify(&tokenized, &mut model);
    }

    Ok(model)
}

/// Builds the word <-> id dictionary over all files. Per-file vocabularies are
/// computed in parallel and merged in file order (keys sorted), so ids are
/// deterministic across runs and thread counts.
pub fn build_vocab(files: &[PathBuf]) -> (HashMap<String, u32>, Vec<String>) {
    let local_vocabs: Vec<Vec<String>> = files
        .par_iter()
        .map(|path| {
            let mut words: Vec<String> = match std::fs::read_to_string(path) {
                Ok(text) => tokenize(&text)
                    .tokens()
                    .iter()
                    .map(|t| t.content.clone())
                    .collect(),
                Err(e) => {
                    eprintln!("skipping {}: {e}", path.display());
                    Vec::new()
                }
            };
            words.sort();
            words.dedup();
            words
        })
        .collect();

    let mut words = HashMap::new();
    let mut strings = Vec::new();
    for file_words in local_vocabs {
        for word in file_words {
            if !words.contains_key(&word) {
                let id = strings.len() as u32;
                strings.push(word.clone());
                words.insert(word, id);
            }
        }
    }
    (words, strings)
}

/// Number of buckets owned by shard `shard_id`: one full position range per
/// word id with `id % num_shards == shard_id`.
pub fn shard_len(n_words: usize, shard_id: usize, num_shards: usize) -> usize {
    (n_words + num_shards - 1 - shard_id) / num_shards * NUMBER_OF_POSITIONS
}

/// Moves shard buckets into the global `(root_id, pos)` layout, forming a
/// TrainingModel. Moves, not copies: peak memory stays at one model's worth.
pub fn assemble_shards(
    words: HashMap<String, u32>,
    strings: Vec<String>,
    mut shards: Vec<Vec<Bucket>>,
    num_shards: usize,
) -> TrainingModel {
    let n_words = strings.len();
    let mut buckets: Vec<Bucket> = Vec::with_capacity(n_words * NUMBER_OF_POSITIONS);
    for root_id in 0..n_words {
        let s = root_id % num_shards;
        let base = (root_id / num_shards) * NUMBER_OF_POSITIONS;
        buckets.extend(
            shards[s][base..base + NUMBER_OF_POSITIONS]
                .iter_mut()
                .map(std::mem::take),
        );
    }
    TrainingModel {
        words,
        strings,
        buckets,
    }
}

/// Parallel training: buckets are sharded by `root_id % num_shards`, each shard
/// is classified by one rayon task over the whole corpus with no locks, then the
/// shards are moved into place to form the TrainingModel. Peak memory is one
/// model's worth (vocab shared read-only + shards partitioning the buckets).
pub fn train_from_dir_parallel(
    dir: impl AsRef<Path>,
    num_shards: usize,
) -> io::Result<TrainingModel> {
    let num_shards = num_shards.max(1);
    let files = list_files(dir)?;
    let (words, strings) = build_vocab(&files);
    let n_words = strings.len();

    let shards: Vec<Vec<Bucket>> = (0..num_shards)
        .into_par_iter()
        .map(|s| {
            let mut shard = vec![Bucket::default(); shard_len(n_words, s, num_shards)];
            for path in &files {
                let Ok(text) = std::fs::read_to_string(path) else {
                    continue;
                };
                let tokenized = tokenize(&text);
                classify_shard(&tokenized, &words, s, num_shards, &mut shard);
            }
            shard
        })
        .collect();

    Ok(assemble_shards(words, strings, shards, num_shards))
}

/// Writes the generation model files (words/strings/buckets/offsets) directly
/// from the shards, streaming bucket by bucket in global `(root_id, pos)` order
/// without ever materializing the assembled TrainingModel — this keeps peak
/// memory at one shard set plus small buffers.
pub fn write_model_from_shards(
    words: &HashMap<String, u32>,
    strings: &[String],
    shards: &mut [Vec<Bucket>],
    num_shards: usize,
    name: &str,
) -> io::Result<()> {
    use crate::model::{DEFAULT_MODEL_PATH, model_path, sorted_entries, write_atomic};
    use std::io::Write;

    std::fs::create_dir_all(DEFAULT_MODEL_PATH)?;
    let config = bincode::config::standard();

    write_atomic(&model_path(name, "words"), |w| {
        bincode::serde::encode_into_std_write(words, w, config)
            .map(|_| ())
            .map_err(io::Error::other)
    })?;
    write_atomic(&model_path(name, "strings"), |w| {
        bincode::serde::encode_into_std_write(strings, w, config)
            .map(|_| ())
            .map_err(io::Error::other)
    })?;

    let n_words = strings.len();
    let mut offsets: Vec<usize> = Vec::with_capacity(n_words * NUMBER_OF_POSITIONS + 1);
    write_atomic(&model_path(name, "buckets"), |w| {
        let mut pos = 0usize;
        for root_id in 0..n_words {
            let s = root_id % num_shards;
            let base = (root_id / num_shards) * NUMBER_OF_POSITIONS;
            for i in 0..NUMBER_OF_POSITIONS {
                offsets.push(pos);
                let bucket = std::mem::take(&mut shards[s][base + i]);
                for (id, count) in sorted_entries(&bucket) {
                    w.write_all(&id.to_le_bytes())?;
                    w.write_all(&count.to_le_bytes())?;
                    pos += 8;
                }
            }
        }
        offsets.push(pos); // end sentinel
        Ok(())
    })?;

    write_atomic(&model_path(name, "buckets_offsets"), |w| {
        bincode::serde::encode_into_std_write(&offsets, w, config)
            .map(|_| ())
            .map_err(io::Error::other)
    })?;

    Ok(())
}

/// Converts a parallel-training checkpoint file into the generation model files
/// under models/<name>_*.bin, streaming the checkpoint value-by-value: only one
/// bucket (HashMap) is in memory at a time while decoding; each shard is kept
/// afterwards as flat 8-byte (follower id, count) pairs plus byte offsets, so
/// peak memory is roughly the flat model size (~pairs x 8B), not the expanded
/// in-memory bucket maps.
///
/// Note: the checkpoint's bucket encoding depends on the `hash-buckets`
/// feature, so build the converter with the same feature the checkpoint was
/// written with.
pub fn convert_checkpoint_to_model(
    checkpoint_path: impl AsRef<Path>,
    name: &str,
) -> io::Result<()> {
    use crate::model::{model_path, sorted_entries, write_atomic, DEFAULT_MODEL_PATH};
    use std::fs::File;
    use std::io::{BufReader, Write};

    let mut reader = BufReader::new(File::open(checkpoint_path)?);
    let config = bincode::config::standard();

    // ParallelCheckpoint { words, strings, shards, num_shards, lines_done },
    // decoded field by field so shards can be freed as they are flattened.
    let words: HashMap<String, u32> =
        bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;
    let strings: Vec<String> =
        bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;
    let n_words = strings.len();
    let shards_len: usize =
        bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;

    // Flatten each shard to raw bytes + per-bucket byte offsets.
    let mut flat_shards: Vec<(Vec<u8>, Vec<usize>)> = Vec::with_capacity(shards_len);
    for s in 0..shards_len {
        let buckets_len: usize =
            bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;
        let mut flat = Vec::new();
        let mut offsets = Vec::with_capacity(buckets_len + 1);
        for _ in 0..buckets_len {
            let bucket: Bucket =
                bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;
            offsets.push(flat.len());
            for (id, count) in sorted_entries(&bucket) {
                flat.extend_from_slice(&id.to_le_bytes());
                flat.extend_from_slice(&count.to_le_bytes());
            }
        }
        offsets.push(flat.len());
        flat_shards.push((flat, offsets));
    }

    let num_shards: usize =
        bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;
    let _lines_done: usize =
        bincode::serde::decode_from_std_read(&mut reader, config).map_err(io::Error::other)?;
    assert_eq!(num_shards, shards_len, "checkpoint shard count mismatch");

    std::fs::create_dir_all(DEFAULT_MODEL_PATH)?;

    write_atomic(&model_path(name, "words"), |w| {
        bincode::serde::encode_into_std_write(&words, w, config)
            .map(|_| ())
            .map_err(io::Error::other)
    })?;
    write_atomic(&model_path(name, "strings"), |w| {
        bincode::serde::encode_into_std_write(&strings, w, config)
            .map(|_| ())
            .map_err(io::Error::other)
    })?;

    // Merge shards into global (root_id, pos) bucket order.
    let mut offsets: Vec<usize> = Vec::with_capacity(n_words * NUMBER_OF_POSITIONS + 1);
    write_atomic(&model_path(name, "buckets"), |w| {
        let mut pos = 0usize;
        for root_id in 0..n_words {
            let s = root_id % num_shards;
            let base = (root_id / num_shards) * NUMBER_OF_POSITIONS;
            let (flat, shard_offsets) = &flat_shards[s];
            for i in 0..NUMBER_OF_POSITIONS {
                offsets.push(pos);
                let (start, end) = (shard_offsets[base + i], shard_offsets[base + i + 1]);
                w.write_all(&flat[start..end])?;
                pos += end - start;
            }
        }
        offsets.push(pos); // end sentinel
        Ok(())
    })?;

    write_atomic(&model_path(name, "buckets_offsets"), |w| {
        bincode::serde::encode_into_std_write(&offsets, w, config)
            .map(|_| ())
            .map_err(io::Error::other)
    })?;

    Ok(())
}
