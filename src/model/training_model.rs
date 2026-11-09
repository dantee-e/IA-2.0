use std::io::{self, Write};

use super::{model_path, write_atomic};
use crate::{
    Model, NUMBER_OF_POSITIONS,
    model::{DEFAULT_MODEL_PATH, read_memmap_lines::LineIndex},
};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Default)]
pub struct TrainingModel {
    // word <-> id dictionary
    pub words: HashMap<String, u32>,
    pub strings: Vec<String>,

    // bucket index -> (follower word id, count)
    pub buckets: Vec<Vec<(u32, u32)>>,
}

impl TrainingModel {
    fn intern(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.words.get(word) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(word.to_owned());
        self.words.insert(word.to_owned(), id);
        self.buckets.extend(vec![Vec::new(); NUMBER_OF_POSITIONS]);
        id
    }

    pub fn bucket_id(&self, key: (u32, u16)) -> usize {
        key.0 as usize * 100 + key.1 as usize
    }

    pub fn insert_or_increase_count(&mut self, root: &str, pos: u16, follower: &str) {
        let root_id = self.intern(root);
        let follower_id = self.intern(follower);
        let bucket = self.bucket_id((root_id, pos));

        let entries = &mut self.buckets[bucket];
        match entries.iter_mut().find(|e| e.0 == follower_id) {
            Some(e) => e.1 += 1,
            None => entries.push((follower_id, 1)),
        }
    }

    /// Words seen `pos` tokens after `root`, with counts. Never modifies the model.
    pub fn followers(&self, root: &str, pos: u16) -> impl Iterator<Item = (&str, u32)> + '_ {
        let bucket = self
            .words
            .get(root)
            .map(|&root_id| self.bucket_id((root_id, pos)))
            .map(|b| self.buckets[b].as_slice())
            .unwrap_or(&[]);

        bucket
            .iter()
            .map(|&(id, count)| (self.strings[id as usize].as_str(), count))
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

        // Buckets: records back to back; offsets[i]..offsets[i + 1] is record i.
        let mut offsets: Vec<usize> = Vec::with_capacity(self.buckets.len() + 1);
        write_atomic(&path_buckets, |w| {
            let mut pos = 0usize;
            for bucket in &self.buckets {
                offsets.push(pos);
                let bytes =
                    bincode::serde::encode_to_vec(bucket, config).map_err(io::Error::other)?;
                w.write_all(&bytes)?;
                pos += bytes.len();
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
