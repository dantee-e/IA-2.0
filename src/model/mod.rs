use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
pub mod persist;

fn id_to_usize(id: (String, usize)) {}

#[derive(Deserialize, Serialize, Default)]
pub struct TrainingModel {
    count: HashMap<(String, usize), Vec<(String, usize)>>,
    finished: bool,
}

impl TrainingModel {
    pub fn insert_or_count(&mut self, root_and_pos: (String, usize), value: String) {
        let entries = self.count.entry(root_and_pos).or_insert_with(Vec::new);

        match entries.iter_mut().find(|e| e.0 == value) {
            Some(e) => e.1 += 1,
            None => entries.push((value, 1)),
        }
    }

    pub fn get(&self, id: (String, usize)) -> Option<&Vec<(String, usize)>> {
        self.count.get(&id)
    }

    pub fn get_entry(&self, id: (String, usize), value: &String) -> Option<usize> {
        Some(self.count.get(&id)?.iter().find(|&v| &v.0 == value)?.1)
    }
}
