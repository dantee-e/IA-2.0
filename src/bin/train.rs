use ai2::{NUMERO_DE_POSICOES, TrainingModel, classify, load_model, save_model, tokenize};

const TRAINING_JSON: &str = "tests/long_texts/gutenberg_v2.jsonl";
const DEFAULT_MODEL_PATH: &str = "models";

#[derive(serde::Deserialize)]
struct JsonLines {
    text: String,
}

fn main() {
    let file = std::fs::File::open(TRAINING_JSON).expect("Could not find json");
    let deserializer = serde_json::Deserializer::from_reader(file);
    let mut model: TrainingModel = TrainingModel::default();

    let run = 0;

    let model_name = format!(
        "{DEFAULT_MODEL_PATH}/model_{}_{}.bin",
        NUMERO_DE_POSICOES, run
    );

    let skip = 1100;

    // while std::fs::exists(model_name.clone()).unwrap() {
    //     println!("Exists model {}", model_name);
    //     model_name = format!(
    //         "{DEFAULT_MODEL_PATH}/model_{}_{}.bin",
    //         NUMERO_DE_POSICOES, run
    //     );
    //     run += 1;
    // }

    if std::fs::exists(model_name.clone()).unwrap() {
        println!("Loading model");
        model = load_model(model_name.clone()).unwrap();
        println!("Model loaded");
    }

    for (i, line) in deserializer.into_iter::<JsonLines>().enumerate() {
        if i < skip {
            println!("Skipping i = {i}");
            continue;
        }
        println!(
            "{}% complete, Entries {}, i={i}/213000",
            i as f32 * 100.0 / 213000.0,
            model.len()
        );
        if i % 100 == 0 {
            match save_model(&model, model_name.clone()) {
                Ok(_) => println!("Saved model."),
                Err(_) => eprintln!("Failed to save model"),
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
    }

    if let Err(e) = save_model(&model, &model_name) {
        eprintln!("Failed to save model: {e}");
    }
}
