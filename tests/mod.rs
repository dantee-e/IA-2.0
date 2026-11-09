#[cfg(test)]
mod tests {
    use ai2::*;
    #[test]
    fn classify_phrase() {
        let text = "To tokenize sentences in Rust, you can use specialized libraries depending on your needs, such as sentence for TTS-style lexing, rust-punkt for unsupervised sentence splitting, or vtext for Unicode-based segmentation. For high-performance NLP tasks involving modern language models (e.g., BERT, SentencePiece), the rust_tokenizers crate is the standard choice, offering support for WordPiece, BPE, and Unigram models.";
        let tokenized_text = tokenize(text);

        let mut training_model = TrainingModel::default();
        println!(
            "Running classify_phrase with token count {}",
            tokenized_text.tokens().len()
        );

        classify(&tokenized_text, &mut training_model);

        let model = training_model.into_model("phrase").unwrap();

        let text = " To tokenize sentences in Rust, you can use";
        let mut tokenized_text = tokenize(text);

        for _ in 0..100 {
            generate(&mut tokenized_text, &model);
        }

        println!("{}", tokenized_text.text());
    }

    const LONG_TEXTS: &str = "tests/long_texts";
    const RESULTS: &str = "tests/results";

    #[test]
    fn classify_beemovie() {
        let beemovie_script =
            std::fs::read_to_string(format!("{LONG_TEXTS}/beemovie.txt")).unwrap();
        let mut tokenized_script = tokenize(&beemovie_script);

        let mut training_model: TrainingModel = TrainingModel::default();

        classify(&tokenized_script, &mut training_model);

        let model = training_model.into_model("beemovie").unwrap();

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

        let mut training_model: TrainingModel = TrainingModel::default();

        classify(&tokenized_script, &mut training_model);

        let model = training_model
            .into_model("the_orneks_and_the_gogos")
            .unwrap();

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

        let mut training_model: TrainingModel = TrainingModel::default();

        classify(&tokenized_script, &mut training_model);
        classify(&tokenized_orneks, &mut training_model);

        let model = training_model
            .into_model("the_orneks_and_the_gogos")
            .unwrap();

        let mut my_tokenized_phrase =
            tokenize("My name is the wind and the wind does as the wind ");

        for _ in 0..100 {
            generate(&mut my_tokenized_phrase, &model);
        }
        println!("{}", my_tokenized_phrase.text());
    }
}
