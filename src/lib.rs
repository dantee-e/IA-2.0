use std::io;
use std::path::Path;

mod classifier;
mod generator;
mod interner;
mod model;
mod tokenizer;

pub use classifier::classify;
pub use generator::generate;
pub use model::TrainingModel;
pub use model::persist::{load_model, save_model};
pub use tokenizer::Mask;
pub use tokenizer::Token;
pub use tokenizer::tokenize;

pub const NUMERO_DE_POSICOES: usize = 100;

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

#[cfg(test)]
mod tests {
    use crate::generator::generate;

    use super::*;
    #[test]
    fn classify_phrase() {
        let text = "To tokenize sentences in Rust, you can use specialized libraries depending on your needs, such as sentence for TTS-style lexing, rust-punkt for unsupervised sentence splitting, or vtext for Unicode-based segmentation. For high-performance NLP tasks involving modern language models (e.g., BERT, SentencePiece), the rust_tokenizers crate is the standard choice, offering support for WordPiece, BPE, and Unigram models.";
        let tokenized_text = tokenize(text);
        classify(&tokenized_text, &mut TrainingModel::default());

        // for token in classified.into_iter() {
        //     println!("{}:", token.0);
        //     for relation in token.1.into_iter() {
        //         println!("  {}:", relation.0);
        //         for word in relation.1 {
        //             println!("      {}: {}", word.0, word.1);
        //         }
        //     }
        // }
    }

    const LONG_TEXTS: &str = "tests/long_texts";
    const RESULTS: &str = "tests/results";

    #[test]
    fn classify_beemovie() {
        let beemovie_script =
            std::fs::read_to_string(format!("{LONG_TEXTS}/beemovie.txt")).unwrap();
        let mut tokenized_script = tokenize(&beemovie_script);

        let mut model: TrainingModel = TrainingModel::default();

        classify(&tokenized_script, &mut model);

        for _ in 0..100 {
            generate(&mut tokenized_script, &model);
        }

        std::fs::write(format!("{RESULTS}/beemovie.txt"), tokenized_script.text()).unwrap()
    }

    #[test]
    fn classify_the_orneks_and_the_gogos() {
        let the_orneks_and_the_gogos =
            std::fs::read_to_string(format!("{LONG_TEXTS}/the_orneks_and_the_gogos.txt")).unwrap();
        let mut tokenized_script = tokenize(&the_orneks_and_the_gogos);

        let mut model: TrainingModel = TrainingModel::default();

        classify(&tokenized_script, &mut model);

        for _ in 0..100 {
            generate(&mut tokenized_script, &model);
        }

        std::fs::write(
            format!("{RESULTS}/the_orneks_and_the_gogos.txt"),
            tokenized_script.text(),
        )
        .unwrap()
    }

    #[test]
    fn classify_and_train_many_books() {
        let beemovie_script =
            std::fs::read_to_string(format!("{LONG_TEXTS}/beemovie.txt")).unwrap();
        let tokenized_script = tokenize(&beemovie_script);

        let the_orneks_and_the_gogos =
            std::fs::read_to_string(format!("{LONG_TEXTS}/the_orneks_and_the_gogos.txt")).unwrap();
        let tokenized_orneks = tokenize(&the_orneks_and_the_gogos);

        let mut model: TrainingModel = TrainingModel::default();

        classify(&tokenized_script, &mut model);
        classify(&tokenized_orneks, &mut model);

        let mut my_tokenized_phrase =
            tokenize("My name is the wind and the wind does as the wind ");

        for _ in 0..100 {
            generate(&mut my_tokenized_phrase, &model);
        }
        println!("{}", my_tokenized_phrase.text());
    }
}
