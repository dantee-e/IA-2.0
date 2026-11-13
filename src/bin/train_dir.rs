use std::env;
use std::time::Instant;

// Trains a model from every file in a directory (e.g. a folder of books) and
// writes it under models/<name>_*.bin.
// Usage: train_dir <dir> <model_name> [num_shards]   (num_shards > 1 = parallel)
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: {} <dir> <model_name> [num_shards]", args[0]);
        std::process::exit(1);
    }
    let num_shards: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let start = Instant::now();
    let model = if num_shards > 1 {
        ai2::train_from_dir_parallel(&args[1], num_shards).expect("could not train from dir")
    } else {
        ai2::train_from_dir(&args[1]).expect("could not train from dir")
    };
    println!(
        "trained ({} shards): {} words, {} buckets in {:.2?}",
        num_shards,
        model.words.len(),
        model.len(),
        start.elapsed()
    );

    let model = model;
    model
        .into_model(&args[2])
        .expect("could not write model");
    println!("wrote model '{}'", args[2]);
}
