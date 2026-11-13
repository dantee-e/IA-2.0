use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::{Bucket, TrainingModel};

pub fn save_training_model(
    model: &TrainingModel,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");

    let mut writer = BufWriter::new(File::create(&tmp)?);
    bincode::serde::encode_into_std_write(model, &mut writer, bincode::config::standard())?;
    writer.flush()?;
    writer.get_ref().sync_all()?; // make sure it hit disk
    drop(writer);

    std::fs::rename(&tmp, path)?; // atomic: readers never see a half-written file
    Ok(())
}

pub fn load_training_model(path: impl AsRef<Path>) -> Result<TrainingModel, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let model = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    Ok(model)
}

/// Checkpoint for parallel (root-sharded) training: vocab plus one bucket array
/// per shard, so saving never has to assemble the full TrainingModel.
#[derive(Deserialize)]
pub struct ParallelCheckpoint {
    pub words: HashMap<String, u32>,
    pub strings: Vec<String>,
    pub shards: Vec<Vec<Bucket>>,
    pub num_shards: usize,
    pub lines_done: usize,
}

pub fn save_parallel_checkpoint(
    path: impl AsRef<Path>,
    words: &HashMap<String, u32>,
    strings: &[String],
    shards: &[Vec<Bucket>],
    num_shards: usize,
    lines_done: usize,
) -> Result<(), Box<dyn Error>> {
    #[derive(Serialize)]
    struct CheckpointRef<'a> {
        words: &'a HashMap<String, u32>,
        strings: &'a [String],
        shards: &'a [Vec<Bucket>],
        num_shards: usize,
        lines_done: usize,
    }

    let path = path.as_ref();
    let tmp = path.with_extension("tmp");

    let mut writer = BufWriter::new(File::create(&tmp)?);
    bincode::serde::encode_into_std_write(
        &CheckpointRef {
            words,
            strings,
            shards,
            num_shards,
            lines_done,
        },
        &mut writer,
        bincode::config::standard(),
    )?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);

    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load_parallel_checkpoint(path: impl AsRef<Path>) -> Result<ParallelCheckpoint, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let cp = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    Ok(cp)
}
