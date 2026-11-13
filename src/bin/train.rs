use std::env;
use std::fs::File;
use std::time::Instant;

use ai2::{
    Bucket, NUMBER_OF_POSITIONS, classify_shard, load_parallel_checkpoint,
    save_parallel_checkpoint, shard_len, tokenize, write_model_from_shards,
};
use hashbrown::HashMap;
use rayon::prelude::*;

const TRAINING_JSON: &str = "tests/long_texts/gutenberg_v2.jsonl";
const DEFAULT_TRAINING_MODEL_PATH: &str = "training_models";
const CHUNK_SIZE: usize = 256;
const CHECKPOINT_EVERY: usize = 2000;

#[derive(serde::Deserialize)]
struct JsonLines {
    text: String,
}

/// Streams the jsonl file in chunks of lines, calling `f(chunk_texts, start_line)`
/// for each chunk. Lines before `start_line` are parsed but skipped (resume).
fn process_in_chunks(limit: usize, start_line: usize, mut f: impl FnMut(Vec<String>, usize)) {
    let file = File::open(TRAINING_JSON).expect("Could not find json");
    let deserializer = serde_json::Deserializer::from_reader(file);

    let mut i = 0usize;
    let mut chunk: Vec<String> = Vec::with_capacity(CHUNK_SIZE);
    for line in deserializer.into_iter::<JsonLines>() {
        if i >= limit {
            break;
        }
        let Ok(l) = line else {
            eprintln!("Parse error at record {i}");
            i += 1;
            continue;
        };
        i += 1;
        if i <= start_line {
            continue;
        }
        chunk.push(l.text);
        if chunk.len() == CHUNK_SIZE {
            f(std::mem::take(&mut chunk), i - CHUNK_SIZE);
        }
    }
    if !chunk.is_empty() {
        let start = i - chunk.len();
        f(chunk, start);
    }
}

/// Pass 0: builds the global vocab streaming the file in chunks. Words get ids
/// in first-encounter order (deterministic, same as sequential training).
fn build_vocab(limit: usize) -> (HashMap<String, u32>, Vec<String>) {
    let mut words = HashMap::new();
    let mut strings = Vec::new();
    process_in_chunks(limit, 0, |chunk, _| {
        let docs: Vec<_> = chunk.par_iter().map(|text| tokenize(text)).collect();
        for doc in &docs {
            for token in doc.tokens() {
                if !words.contains_key(&token.content) {
                    let id = strings.len() as u32;
                    strings.push(token.content.clone());
                    words.insert(token.content.clone(), id);
                }
            }
        }
    });
    (words, strings)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let run: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let train_with: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10000);
    let num_shards: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8).max(1);

    std::fs::create_dir_all(DEFAULT_TRAINING_MODEL_PATH).expect("could not create checkpoint dir");
    let checkpoint_path = format!(
        "{DEFAULT_TRAINING_MODEL_PATH}/model_{NUMBER_OF_POSITIONS}_{run}_par.bin"
    );
    let final_name = format!("model_{NUMBER_OF_POSITIONS}_{run}");

    let (words, strings, mut shards, start_line) = match load_parallel_checkpoint(&checkpoint_path) {
        Ok(cp) if cp.num_shards == num_shards => {
            println!(
                "resumed checkpoint at line {}/{} ({} shards)",
                cp.lines_done, train_with, cp.num_shards
            );
            (cp.words, cp.strings, cp.shards, cp.lines_done)
        }
        _ => {
            println!("pass 0: building vocab over {train_with} lines...");
            let t = Instant::now();
            let (words, strings) = build_vocab(train_with);
            let n_words = strings.len();
            println!("vocab: {n_words} words in {:.2?}", t.elapsed());
            let shards = (0..num_shards)
                .map(|s| vec![Bucket::default(); shard_len(n_words, s, num_shards)])
                .collect();
            (words, strings, shards, 0usize)
        }
    };
    let n_words = strings.len();

    let mut lines_done = start_line;
    let mut checkpoint_mark = start_line / CHECKPOINT_EVERY;
    let t = Instant::now();

    process_in_chunks(train_with, start_line, |chunk, start| {
        let docs: Vec<_> = chunk.par_iter().map(|text| tokenize(text)).collect();
        (0..num_shards)
            .into_par_iter()
            .zip(shards.par_iter_mut())
            .for_each(|(s, shard)| {
                for doc in &docs {
                    classify_shard(doc, &words, s, num_shards, shard);
                }
            });
        lines_done = start + docs.len();

        println!(
            "{:.2}% complete, words {n_words}, i={lines_done}/{train_with}",
            lines_done as f32 * 100.0 / train_with as f32,
        );

        if lines_done / CHECKPOINT_EVERY > checkpoint_mark {
            checkpoint_mark = lines_done / CHECKPOINT_EVERY;
            let t = Instant::now();
            match save_parallel_checkpoint(
                &checkpoint_path,
                &words,
                &strings,
                &shards,
                num_shards,
                lines_done,
            ) {
                Ok(_) => println!(
                    "saved checkpoint at line {lines_done} in {:.2?}",
                    t.elapsed()
                ),
                Err(e) => eprintln!("Failed to save checkpoint: {e}"),
            }
        }
    });

    println!("classified {lines_done} lines in {:.2?}", t.elapsed());

    // Final checkpoint, then assemble and write the generation model.
    if let Err(e) = save_parallel_checkpoint(
        &checkpoint_path,
        &words,
        &strings,
        &shards,
        num_shards,
        lines_done,
    ) {
        eprintln!("Failed to save final checkpoint: {e}");
    }

    // Write the model straight from the shards: assembling a full TrainingModel
    // first would need a second bucket-header array on top of the shards and can
    // OOM on large models.
    let t = Instant::now();
    match write_model_from_shards(&words, &strings, &mut shards, num_shards, &final_name) {
        Ok(_) => println!(
            "Wrote final model '{final_name}' in {:.2?}",
            t.elapsed()
        ),
        Err(e) => {
            eprintln!("Failed to write final model '{final_name}': {e}");
            std::process::exit(1);
        }
    }
}
