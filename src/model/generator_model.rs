use super::{model_path, read_bincode};
use crate::{NUMBER_OF_POSITIONS, model::read_memmap_lines::LineIndex};
use hashbrown::HashMap;
use std::io;

/// Decay of a context word's importance with its distance from the end of the
/// text (pos 0 = most recent token): halves every HALF_LIFE positions.
const HALF_LIFE: f32 = 2.0;
/// Periodic modulation on top of the decay, in positions. The most recent words
/// keep the highest weight overall (the envelope), but importance oscillates
/// with this period so structure repeats at clause-like intervals.
const OSCILLATION_PERIOD: f32 = 8.0;
/// Amplitude of the oscillation, in [0, 1]. 0 disables it.
const OSCILLATION_AMPLITUDE: f32 = 0.5;

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
            weights: std::array::from_fn(|i| {
                let pos = i as u16;
                Model::position_weight(pos) * Model::oscillation(pos)
            }),
        }
    }

    fn bucket_id(key: (u32, u16)) -> usize {
        key.0 as usize * NUMBER_OF_POSITIONS + key.1 as usize
    }

    pub fn key_from_bucket_id(id: usize) -> (u32, u16) {
        (
            (id / NUMBER_OF_POSITIONS) as u32,
            (id % NUMBER_OF_POSITIONS) as u16,
        )
    }

    fn position_weight(pos: u16) -> f32 {
        0.5f32.powf(pos as f32 / HALF_LIFE)
    }

    fn oscillation(pos: u16) -> f32 {
        1.0 + OSCILLATION_AMPLITUDE
            * (std::f32::consts::TAU * pos as f32 / OSCILLATION_PERIOD).cos()
    }

    fn weigh_count_with_position(&self, count: u32, pos: u16) -> f32 {
        count as f32 * self.weights[pos as usize]
    }

    pub fn followers(&self, root: &str, pos: u16) -> impl Iterator<Item = (&str, f32)> + '_ {
        let bytes = match self.words.get(root) {
            Some(&root_id) => self
                .buckets
                .record(Self::bucket_id((root_id, pos)))
                .unwrap_or(&[]),
            None => &[],
        };

        bytes.chunks_exact(8).map(move |chunk| {
            let id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
            let count = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
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
            weights: std::array::from_fn(|i| {
                let pos = i as u16;
                Model::position_weight(pos) * Model::oscillation(pos)
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_id_round_trip() {
        for key in [
            (0u32, 0u16),
            (1, 0),
            (1, 299),
            (5, 150),
            (164_445, 299),
            (u32::MAX / 2, 42),
        ] {
            assert_eq!(Model::key_from_bucket_id(Model::bucket_id(key)), key);
        }
    }

    #[test]
    fn weights_favor_recent_positions_and_oscillate() {
        let w = |p: u16| Model::position_weight(p) * Model::oscillation(p);
        // overall envelope decays: pos 0 must beat pos 100
        assert!(w(0) > w(100));
        // oscillation: weight is not monotonic between nearby positions
        let diffs: Vec<bool> = (1..32).map(|p| w(p) > w(p - 1)).collect();
        assert!(diffs.iter().any(|&d| d) && diffs.iter().any(|&d| !d));
        // weights never go negative, and the envelope underflow only at the
        // extreme tail (f32 subnormals), so the weighted sum stays positive
        assert!((0..NUMBER_OF_POSITIONS).all(|p| w(p as u16) >= 0.0));
        assert!((0..100).all(|p| w(p as u16) > 0.0));
    }
}
