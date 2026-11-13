use std::env;
use std::time::Instant;

// Converts a parallel-training checkpoint (training_models/<...>_par.bin) into
// the generation model files models/<name>_*.bin, using low memory (the
// checkpoint is streamed bucket-by-bucket, never fully in RAM).
//
// Usage: convert <checkpoint_path> <model_name>
//
// The checkpoint's bucket encoding depends on the `hash-buckets` cargo feature,
// so build with the same feature the checkpoint was written with.
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <checkpoint_path> <model_name>", args[0]);
        std::process::exit(1);
    }

    let start = Instant::now();
    ai2::convert_checkpoint_to_model(&args[1], &args[2]).expect("conversion failed");
    println!(
        "converted '{}' into model '{}' in {:.2?}",
        args[1],
        args[2],
        start.elapsed()
    );
}
