use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::Path,
};

use super::Model;

pub fn save_model(model: &Model, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    bincode::serde::encode_into_std_write(model, &mut writer, bincode::config::standard())?;
    writer.flush()?;
    Ok(())
}

pub fn load_model(path: impl AsRef<Path>) -> Result<Model, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let model = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;
    Ok(model)
}
