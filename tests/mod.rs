#[cfg(test)]
mod tests {
    use ai2::*;

    #[test]
    fn followers_respect_position_after_save() {
        // Regression: bucket_id used a hardcoded *100 while the generator reads
        // *NUMBER_OF_POSITIONS, so saved buckets landed in the wrong slots.
        let mut training_model = TrainingModel::default();
        classify(&tokenize("the cat sat on the mat"), &mut training_model);
        let model = training_model.into_model("unit_followers").unwrap();

        let at0: Vec<&str> = model.followers("cat", 0).map(|(w, _)| w).collect();
        let at1: Vec<&str> = model.followers("cat", 1).map(|(w, _)| w).collect();
        let at2: Vec<&str> = model.followers("cat", 2).map(|(w, _)| w).collect();

        assert_eq!(at0, vec!["sat"]);
        assert_eq!(at1, vec!["on"]);
        assert_eq!(at2, vec!["the"]);

        // unknown root yields nothing
        assert!(model.followers("nonexistent", 0).next().is_none());
    }

    #[test]
    fn parallel_matches_sequential() {
        let dir = std::env::temp_dir().join("ai2_par_test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.txt"),
            "the cat sat on the mat the cat sat on the dog the cat ran",
        )
        .unwrap();

        let seq = train_from_dir(&dir).unwrap();
        let par = train_from_dir_parallel(&dir, 4).unwrap();

        assert_eq!(seq.strings.len(), par.strings.len());
        assert_eq!(seq.buckets.len(), par.buckets.len());
        for (word, _) in &seq.words {
            for pos in 0..NUMBER_OF_POSITIONS as u16 {
                let mut a = seq.followers(word, pos);
                let mut b = par.followers(word, pos);
                a.sort();
                b.sort();
                assert_eq!(a, b, "mismatch for '{word}' at pos {pos}");
            }
        }
    }

    #[test]
    fn write_model_from_shards_matches_into_model() {
        let dir = std::env::temp_dir().join("ai2_par_test2");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("a.txt"),
            "the cat sat on the mat the cat sat on the dog the cat ran far away",
        )
        .unwrap();

        let par = train_from_dir_parallel(&dir, 3).unwrap();
        let via_into_model = par.into_model("unit_shard_assemble").unwrap();

        // rebuild shards and write through the streaming path
        let par2 = train_from_dir_parallel(&dir, 3).unwrap();
        let TrainingModel {
            words,
            strings,
            buckets,
        } = par2;
        // re-shard the assembled buckets
        let num_shards = 3;
        let n_words = strings.len();
        let mut shards: Vec<Vec<ai2::Bucket>> = (0..num_shards)
            .map(|s| vec![ai2::Bucket::default(); ai2::shard_len(n_words, s, num_shards)])
            .collect();
        for root_id in 0..n_words {
            let s = root_id % num_shards;
            let base = (root_id / num_shards) * NUMBER_OF_POSITIONS;
            for i in 0..NUMBER_OF_POSITIONS {
                shards[s][base + i] = buckets[root_id * NUMBER_OF_POSITIONS + i].clone();
            }
        }
        ai2::write_model_from_shards(&words, &strings, &mut shards, num_shards, "unit_shard_stream")
            .unwrap();
        let via_stream = Model::load("unit_shard_stream").unwrap();

        for (word, _) in &via_into_model.words {
            for pos in 0..NUMBER_OF_POSITIONS as u16 {
                let mut a = via_into_model.followers(word, pos).collect::<Vec<_>>();
                let mut b = via_stream.followers(word, pos).collect::<Vec<_>>();
                a.sort_by(|x, y| x.0.cmp(y.0));
                b.sort_by(|x, y| x.0.cmp(y.0));
                assert_eq!(a, b, "mismatch for '{word}' at pos {pos}");
            }
        }
    }

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
