use ai2::{Model, NUMERO_DE_POSICOES, classify, save_model, tokenize};
use hashbrown::HashMap;

const TRAINING_JSON: &str = "tests/long_texts/gutenberg_v2.jsonl";
const DEFAULT_MODEL_PATH: &str = "models";

#[derive(serde::Deserialize)]
struct JsonLines {
    text: String,
}

fn main() {
    let file = std::fs::File::open(TRAINING_JSON).expect("Could not find json");
    let deserializer = serde_json::Deserializer::from_reader(file);
    let mut model: Model = HashMap::new();

    let model_name = format!("{DEFAULT_MODEL_PATH}/model_{}.bin", NUMERO_DE_POSICOES,);

    for (i, line) in deserializer.into_iter::<JsonLines>().enumerate() {
        println!("{}% complete,  i={i}/213000", i as f32 * 100.0 / 213000.0);
        println!("Entries in model: {}", model.keys().len());
        if i % 10 == 0 {
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
