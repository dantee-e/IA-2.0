mod generator_model;
pub mod persist;
mod read_memmap_lines;
mod training_model;
pub use generator_model::Model;
use serde::de::DeserializeOwned;
pub(crate) use training_model::{increase_count, sorted_entries};
pub use training_model::{Bucket, TrainingModel};

use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
    path::PathBuf,
};

pub(crate) const DEFAULT_MODEL_PATH: &str = "models";

pub(crate) fn model_path(name: &str, kind: &str) -> PathBuf {
    PathBuf::from(format!("{DEFAULT_MODEL_PATH}/{name}_{kind}.bin"))
}

/// Writes to `<path>.tmp`, flushes, fsyncs, then atomically renames over `path`.
pub(crate) fn write_atomic(
    path: &PathBuf,
    write: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);

    let mut writer = BufWriter::new(File::create(&tmp)?);
    write(&mut writer)?;

    // into_inner() flushes the buffer and hands back the File
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);

    std::fs::rename(&tmp, path)
}

pub(super) fn read_bincode<T: DeserializeOwned>(path: &PathBuf) -> io::Result<T> {
    let mut reader = BufReader::new(File::open(path)?);
    bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
        .map_err(io::Error::other)
}
