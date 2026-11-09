use std::time;
use std::{
    env,
    io::{self, Read, Write},
    process,
};

use ai2::generator::{GeneratorTimer, generate_with_timer};
use ai2::{Model, NUMBER_OF_POSITIONS, tokenize};

fn load_model(model_name: &str) -> Option<Model> {
    match Model::load(model_name) {
        Ok(m) => Some(m),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            eprintln!("no saved model '{model_name}'");
            None
        }
        Err(e) => {
            eprintln!("could not load '{model_name}' ({e})");
            None
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    #[allow(clippy::len_zero)]
    if args.len() < 1 || args.len() > 3 {
        eprintln!("usage: {} <num_tokens> [model_name] < prompt.txt", args[0]);
        process::exit(1);
    }

    let num_tokens: usize = args[1].parse().unwrap_or_else(|_| {
        eprintln!("error: '{}' is not a valid number of tokens", args[1]);
        process::exit(1);
    });

    let default_name = format!("model_{NUMBER_OF_POSITIONS}_0");
    let model_name = args.get(2).map(String::as_str).unwrap_or(&default_name);

    let mut prompt = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut prompt) {
        eprintln!("error reading stdin: {e}");
        process::exit(1);
    }

    let start = time::Instant::now();
    eprintln!("loading model '{model_name}'...");

    let model = load_model(model_name).unwrap();

    let elapsed = start.elapsed();
    println!("Time to load model was {:.2?}", elapsed);

    let mut tokenized = tokenize(&prompt);

    let start = time::Instant::now();
    println!("Generating");

    let mut timer = GeneratorTimer::default();

    for _ in 0..num_tokens {
        generate_with_timer(&mut tokenized, &model, &mut timer);
    }
    let elapsed = start.elapsed();
    println!("Time to generate {num_tokens} tokens was {:.2?}", elapsed);

    println!(
        "Time accessing_storage: {:.2?}\nTime choosing candidate: {:.2?}",
        timer.accessing_storage, timer.choosing_candidate
    );

    let mut out = io::stdout().lock();
    if let Err(e) = writeln!(out, "{}", tokenized.text())
        && e.kind() != io::ErrorKind::BrokenPipe
    {
        eprintln!("error writing output: {e}");
        process::exit(1);
    }
}
