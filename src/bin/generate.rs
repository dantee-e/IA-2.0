use std::{
    env,
    io::{self, Read, Write},
    path::Path,
    process,
};

use ai2::{Model, NUMERO_DE_POSICOES, generate, load_model, save_model, tokenize, train_from_dir};

const DEFAULT_TRAINING_DIR: &str = "tests/long_texts";
const DEFAULT_MODEL_PATH: &str = "models";

fn load_or_train(model_path: &str, training_dir: &str) -> Model {
    if Path::new(model_path).exists() {
        eprintln!("loading model from {model_path}...");
        match load_model(model_path) {
            Ok(m) => return m,
            Err(e) => eprintln!("could not load '{model_path}' ({e}), retraining..."),
        }
    }

    eprintln!("training from {training_dir}...");
    let model = match train_from_dir(training_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error reading '{training_dir}': {e}");
            process::exit(1);
        }
    };

    match save_model(&model, model_path) {
        Ok(()) => eprintln!("saved model to {model_path}"),
        Err(e) => eprintln!("warning: could not save model to '{model_path}': {e}"),
    }

    model
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.len() > 4 {
        eprintln!(
            "usage: {} <num_tokens> [training_dir] [model_path] < prompt.txt",
            args[0]
        );
        process::exit(1);
    }

    let num_tokens: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("error: '{}' is not a valid number of tokens", args[1]);
        process::exit(1);
    });

    let training_dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or(DEFAULT_TRAINING_DIR);
    let model_name = format!("{DEFAULT_MODEL_PATH}/model_{}_0.bin", NUMERO_DE_POSICOES,);
    let model_path = args.get(3).map(String::as_str).unwrap_or(&model_name);

    let mut prompt = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut prompt) {
        eprintln!("error reading stdin: {e}");
        process::exit(1);
    }

    let model = load_or_train(model_path, training_dir);

    let mut tokenized = tokenize(&prompt);
    for _ in 0..num_tokens {
        generate(&mut tokenized, &model);
    }

    let mut out = io::stdout().lock();
    if let Err(e) = writeln!(out, "{}", tokenized.text())
        && e.kind() != io::ErrorKind::BrokenPipe
    {
        eprintln!("error writing output: {e}");
        process::exit(1);
    }
}
