pub mod persist;
mod read_memmap_lines;

use bincode::config;
use hashbrown::HashMap;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::{NUMERO_DE_POSICOES, model::read_memmap_lines::LineIndex};

#[derive(Deserialize, Serialize, Default)]
pub struct TrainingModel {
    // word <-> id dictionary
    words: HashMap<String, u32>,
    strings: Vec<String>,

    // bucket index -> (follower word id, count)
    buckets: Vec<Vec<(u32, u32)>>,
}

impl TrainingModel {
    fn intern(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.words.get(word) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(word.to_owned());
        self.words.insert(word.to_owned(), id);
        self.buckets.extend(vec![Vec::new(); NUMERO_DE_POSICOES]);
        id
    }

    fn bucket_id(&self, key: (u32, u16)) -> usize {
        key.0 as usize * 100 + key.1 as usize
    }

    pub fn insert_or_count(&mut self, root: &str, pos: u16, follower: &str) {
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
}

pub struct Model {
    words: HashMap<String, u32>,
    strings: Vec<String>,
    buckets: LineIndex,
}

impl Model {
    fn bucket_id(&self, key: (u32, u16)) -> usize {
        key.0 as usize * 100 + key.1 as usize
    }

    pub fn read_bucket_from_key(&self, key: (u32, u16)) -> Option<Vec<(u32, u32)>> {
        let offset = self.bucket_id(key);
        let config = bincode::config::standard();

        let bytes = self.buckets.line(offset)?;

        let (bucket, _bytes_read): (Vec<(u32, u32)>, usize) =
            bincode::decode_from_slice(bytes, config).ok()?;

        Some(bucket)
    }
}
