use std::io;
use std::path::Path;

mod classifier;
pub mod generator;
mod model;
mod tokenizer;

pub use classifier::classify;
pub use generator::generate;
pub use model::persist::{load_training_model, save_training_model};
pub use model::{Model, TrainingModel};
pub use tokenizer::Mask;
pub use tokenizer::Token;
pub use tokenizer::tokenize;

pub const NUMBER_OF_POSITIONS: usize = 300;

pub fn train_from_dir(dir: impl AsRef<Path>) -> io::Result<TrainingModel> {
    let mut model: TrainingModel = TrainingModel::default();

    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();

        if !path.is_file() {
            continue;
        }

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
