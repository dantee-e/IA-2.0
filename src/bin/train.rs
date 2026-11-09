use ai2::{
    NUMBER_OF_POSITIONS, TrainingModel, classify, load_training_model, save_training_model,
    tokenize,
};

const TRAINING_JSON: &str = "tests/long_texts/gutenberg_v2.jsonl";
const DEFAULT_TRAINING_MODEL_PATH: &str = "training_models";

#[derive(serde::Deserialize)]
struct JsonLines {
    text: String,
}

fn main() {
    let file = std::fs::File::open(TRAINING_JSON).expect("Could not find json");
    let deserializer = serde_json::Deserializer::from_reader(file);

    let run = 0;
    let train_with = 10000;

    std::fs::create_dir_all(DEFAULT_TRAINING_MODEL_PATH).expect("could not create checkpoint dir");

    let checkpoint_path =
        format!("{DEFAULT_TRAINING_MODEL_PATH}/model_{NUMBER_OF_POSITIONS}_{run}.bin");
    let final_name = format!("model_{NUMBER_OF_POSITIONS}_{run}");

    let mut model = if std::fs::exists(&checkpoint_path).unwrap() {
        println!("Loading checkpoint");
        let m = load_training_model(checkpoint_path.clone()).expect("could not load checkpoint");
        println!("Checkpoint loaded");
        m
    } else {
        TrainingModel::default()
    };

    for (i, line) in deserializer.into_iter::<JsonLines>().enumerate() {
        println!(
            "{:.2}% complete, entries {}, i={i}/{train_with}",
            i as f32 * 100.0 / train_with as f32,
            model.len()
        );

        if i % 100 == 0 {
            match save_training_model(&model, checkpoint_path.clone()) {
                Ok(_) => println!("Saved checkpoint."),
                Err(e) => eprintln!("Failed to save checkpoint: {e}"),
            }
        }

        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Parse error at record {i}: {e}");
                continue;
            }
        };

        classify(&tokenize(&line.text), &mut model);

        if i == train_with {
            break;
        }
    }

    // Final checkpoint, so training can be resumed or extended later.
    if let Err(e) = save_training_model(&model, checkpoint_path.clone()) {
        eprintln!("Failed to save checkpoint: {e}");
    }

    // Consumes the TrainingModel and writes words/strings/buckets/offsets under models/.
    match model.into_model(&final_name) {
        Ok(_) => println!("Wrote final model '{final_name}'"),
        Err(e) => {
            eprintln!("Failed to write final model '{final_name}': {e}");
            std::process::exit(1);
        }
    }
}
