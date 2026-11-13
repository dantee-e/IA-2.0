use std::io::{self, Write};

use super::{model_path, write_atomic};
use crate::{
    Model, NUMBER_OF_POSITIONS,
    model::{DEFAULT_MODEL_PATH, read_memmap_lines::LineIndex},
};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

// Bucket storage used while training. Toggle with the `hash-buckets` feature;
// both are serialized through `into_model` into the same on-disk format.
#[cfg(not(feature = "hash-buckets"))]
pub type Bucket = Vec<(u32, u32)>;
#[cfg(feature = "hash-buckets")]
pub type Bucket = HashMap<u32, u32>;

#[cfg(not(feature = "hash-buckets"))]
pub(crate) fn increase_count(bucket: &mut Bucket, follower_id: u32) {
    match bucket.iter_mut().find(|e| e.0 == follower_id) {
        Some(e) => e.1 += 1,
        None => bucket.push((follower_id, 1)),
    }
}

#[cfg(feature = "hash-buckets")]
pub(crate) fn increase_count(bucket: &mut Bucket, follower_id: u32) {
    *bucket.entry(follower_id).or_insert(0) += 1;
}

/// Bucket entries as a sorted vec of (follower id, count), for writing/reading.
pub(crate) fn sorted_entries(bucket: &Bucket) -> Vec<(u32, u32)> {
    #[cfg(not(feature = "hash-buckets"))]
    let mut entries = bucket.clone();
    #[cfg(feature = "hash-buckets")]
    let mut entries: Vec<(u32, u32)> = bucket.iter().map(|(&id, &c)| (id, c)).collect();
    entries.sort_unstable();
    entries
}

#[derive(Deserialize, Serialize, Default)]
pub struct TrainingModel {
    // word <-> id dictionary
    pub words: HashMap<String, u32>,
    pub strings: Vec<String>,

    // bucket index -> follower counts. bucket_id((root_id, pos)) indexes this.
    pub buckets: Vec<Bucket>,
}

impl TrainingModel {
    fn intern(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.words.get(word) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(word.to_owned());
        self.words.insert(word.to_owned(), id);
        self.buckets
            .extend(std::iter::repeat_n(Bucket::default(), NUMBER_OF_POSITIONS));
        id
    }

    pub fn bucket_id(&self, key: (u32, u16)) -> usize {
        key.0 as usize * NUMBER_OF_POSITIONS + key.1 as usize
    }

    pub fn insert_or_increase_count(&mut self, root: &str, pos: u16, follower: &str) {
        let root_id = self.intern(root);
        let follower_id = self.intern(follower);
        let bucket = self.bucket_id((root_id, pos));

        increase_count(&mut self.buckets[bucket], follower_id);
    }

    /// Words seen `pos` tokens after `root`, with counts. Never modifies the model.
    pub fn followers(&self, root: &str, pos: u16) -> Vec<(&str, u32)> {
        let Some(&root_id) = self.words.get(root) else {
            return Vec::new();
        };
        sorted_entries(&self.buckets[self.bucket_id((root_id, pos))])
            .into_iter()
            .map(|(id, count)| (self.strings[id as usize].as_str(), count))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn into_model(self, name: &str) -> io::Result<Model> {
        std::fs::create_dir_all(DEFAULT_MODEL_PATH)?;

        let path_words = model_path(name, "words");
        let path_strings = model_path(name, "strings");
        let path_buckets = model_path(name, "buckets");
        let path_offsets = model_path(name, "buckets_offsets");

        let config = bincode::config::standard();

        write_atomic(&path_words, |w| {
            bincode::serde::encode_into_std_write(&self.words, w, config)
                .map(|_| ())
                .map_err(io::Error::other)
        })?;

        write_atomic(&path_strings, |w| {
            bincode::serde::encode_into_std_write(&self.strings, w, config)
                .map(|_| ())
                .map_err(io::Error::other)
        })?;

        // Buckets: flat records of (follower id: u32, count: u32) LE pairs, 8 bytes
        // each, back to back; offsets[i]..offsets[i + 1] is record i in bytes.
        let mut offsets: Vec<usize> = Vec::with_capacity(self.buckets.len() + 1);
        write_atomic(&path_buckets, |w| {
            let mut pos = 0usize;
            for bucket in &self.buckets {
                offsets.push(pos);
                for (id, count) in sorted_entries(bucket) {
                    w.write_all(&id.to_le_bytes())?;
                    w.write_all(&count.to_le_bytes())?;
                    pos += 8;
                }
            }
            offsets.push(pos); // end sentinel
            Ok(())
        })?;

        write_atomic(&path_offsets, |w| {
            bincode::serde::encode_into_std_write(&offsets, w, config)
                .map(|_| ())
                .map_err(io::Error::other)
        })?;

        Ok(Model::new(
            self.words,
            self.strings,
            LineIndex::new(path_buckets, offsets)?,
        ))
    }
}
