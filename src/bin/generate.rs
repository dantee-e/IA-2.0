use std::{
    env,
    io::{self, Read, Write},
    process,
};

// Replace `your_crate` with the name in your Cargo.toml
use ai2::{generate, tokenize, train_from_dir};

const DEFAULT_TRAINING_DIR: &str = "tests/long_texts";

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "usage: {} <num_tokens> [training_dir] < prompt.txt",
            args[0]
        );
        process::exit(1);
    }

    let num_tokens: usize = match args[1].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("error: '{}' is not a valid number of tokens", args[1]);
            process::exit(1);
        }
    };

    let training_dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or(DEFAULT_TRAINING_DIR);

    // Read the prompt from stdin
    let mut prompt = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut prompt) {
        eprintln!("error reading stdin: {e}");
        process::exit(1);
    }

    // Progress goes to stderr so stdout contains only the result
    eprintln!("training from {training_dir}...");
    let model = match train_from_dir(training_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error reading '{training_dir}': {e}");
            process::exit(1);
        }
    };

    let mut tokenized = tokenize(&prompt);

    for _ in 0..num_tokens {
        generate(&mut tokenized, &model);
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    if let Err(e) = writeln!(out, "{}", tokenized.text()) {
        // Ignore broken pipe (e.g. `| head`), report anything else
        if e.kind() != io::ErrorKind::BrokenPipe {
            eprintln!("error writing output: {e}");
            process::exit(1);
        }
    }
}
