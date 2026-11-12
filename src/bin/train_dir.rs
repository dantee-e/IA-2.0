use std::env;

// Trains a model from every file in a directory (e.g. a folder of books) and
// writes it under models/<name>_*.bin. Usage: train_dir <dir> <model_name>
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <dir> <model_name>", args[0]);
        std::process::exit(1);
    }

    let model = ai2::train_from_dir(&args[1]).expect("could not train from dir");
    println!("trained: {} words, {} buckets", model.words.len(), model.len());

    model
        .into_model(&args[2])
        .expect("could not write model");
    println!("wrote model '{}'", args[2]);
}
