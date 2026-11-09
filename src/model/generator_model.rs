use super::{model_path, read_bincode};
use crate::{NUMBER_OF_POSITIONS, model::read_memmap_lines::LineIndex};
use hashbrown::HashMap;
use std::io;

pub struct Model {
    pub words: HashMap<String, u32>,
    pub strings: Vec<String>,
    pub buckets: LineIndex,
    pub weights: [f32; NUMBER_OF_POSITIONS],
}

impl Model {
    pub fn new(words: HashMap<String, u32>, strings: Vec<String>, buckets: LineIndex) -> Self {
        Self {
            words,
            strings,
            buckets,
            weights: std::array::from_fn(|i| Model::position_weight(i as u16)),
        }
    }
    fn bucket_id(&self, key: (u32, u16)) -> usize {
        key.0 as usize * NUMBER_OF_POSITIONS + key.1 as usize
    }

    pub fn key_from_bucket_id(&self, id: usize) -> (u32, u16) {
        let root_id: u32 = id as u32 / 100 * 100;
        (root_id, (id - root_id as usize) as u16)
    }

    pub fn read_bucket_from_key(&self, key: (u32, u16)) -> Option<Vec<(u32, u32)>> {
        let offset = self.bucket_id(key);

        let config = bincode::config::standard();

        let bytes = self.buckets.record(offset)?;

        let (bucket, _bytes_read): (Vec<(u32, u32)>, usize) =
            bincode::decode_from_slice(bytes, config).ok()?;

        Some(bucket)
    }

    pub fn read_bucket_from_id(&self, id: usize) -> Option<Vec<(u32, u32)>> {
        let config = bincode::config::standard();
        let bytes = self.buckets.record(id)?;

        let (bucket, _bytes_read): (Vec<(u32, u32)>, usize) =
            bincode::decode_from_slice(bytes, config).ok()?;

        Some(bucket)
    }

    fn position_weight(pos: u16) -> f32 {
        const HALF_LIFE: f32 = 2.0;
        0.5f32.powf(pos as f32 / HALF_LIFE)
    }

    fn weigh_count_with_position(&self, count: u32, pos: u16) -> f32 {
        count as f32 * self.weights[pos as usize]
    }

    pub fn followers(&self, root: &str, pos: u16) -> impl Iterator<Item = (&str, f32)> + '_ {
        let bucket = self
            .words
            .get(root)
            .and_then(|root_id| self.read_bucket_from_key((*root_id, pos)))
            .unwrap_or_default();

        bucket.into_iter().map(move |(id, count)| {
            (
                self.strings[id as usize].as_str(),
                self.weigh_count_with_position(count, pos),
            )
        })
    }

    pub fn words(&self) -> &HashMap<String, u32> {
        &self.words
    }

    pub fn strings(&self) -> &Vec<String> {
        &self.strings
    }

    pub fn load(name: &str) -> io::Result<Model> {
        let words: HashMap<String, u32> = read_bincode(&model_path(name, "words"))?;
        let strings: Vec<String> = read_bincode(&model_path(name, "strings"))?;
        let offsets: Vec<usize> = read_bincode(&model_path(name, "buckets_offsets"))?;

        let buckets = LineIndex::new(model_path(name, "buckets"), offsets)?;

        Ok(Model {
            words,
            strings,
            buckets,
            weights: std::array::from_fn(|i| Model::position_weight(i as u16)),
        })
    }
}
