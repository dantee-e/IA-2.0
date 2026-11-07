use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use super::TrainingModel;

pub fn save_model(model: &TrainingModel, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
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

pub fn load_model(path: impl AsRef<Path>) -> Result<TrainingModel, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let model = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    Ok(model)
}
