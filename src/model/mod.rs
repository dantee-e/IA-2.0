use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
pub mod persist;

#[derive(Deserialize, Serialize, Default)]
pub struct TrainingModel {
    count: HashMap<u32, Vec<(String, usize)>>,
    finished: bool,
    ids: HashMap<(String, u16), u32>,
}

impl TrainingModel {
    pub fn insert_or_count(&mut self, root_and_pos: (String, u16), value: String) {
        let id = self.id_from_key(root_and_pos);
        let entries = self.count.entry(id).or_default();

        match entries.iter_mut().find(|e| e.0 == value) {
            Some(e) => e.1 += 1,
            None => entries.push((value, 1)),
        }
    }

    pub fn get_from_key(&self, key: &(String, u16)) -> Option<&Vec<(String, usize)>> {
        self.count.get(&self.id_from_key_option(key)?)
    }

    pub fn get_from_id(&self, id: u32) -> Option<&Vec<(String, usize)>> {
        self.count.get(&id)
    }

    pub fn get_count_from_key(&mut self, key: (String, u16), value: &String) -> Option<usize> {
        let id = self.id_from_key(key);
        Some(self.count.get(&id)?.iter().find(|&v| &v.0 == value)?.1)
    }

    pub fn len(&self) -> usize {
        self.count.len()
    }

    pub fn is_empty(&self) -> bool {
        self.count.is_empty()
    }

    pub fn id_from_key(&mut self, key: (String, u16)) -> u32 {
        if let Some(id) = self.ids.get(&key) {
            *id
        } else {
            let new_id = self.ids.len() as u32;
            self.ids.insert(key, new_id);
            new_id
        }
    }

    pub fn id_from_key_option(&self, key: &(String, u16)) -> Option<u32> {
        self.ids.get(key).map(|a| *a)
    }
}
