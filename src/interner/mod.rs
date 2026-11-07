use std::collections::HashMap;

/// Maps each distinct token string to a u32 id, and back.
#[derive(Default)]
pub struct Interner {
    ids: HashMap<String, u32>,
    strings: Vec<String>,
}

impl Interner {
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.strings.len() as u32;
        self.strings.push(s.to_owned());
        self.ids.insert(s.to_owned(), id);
        id
    }

    pub fn get(&self, s: &str) -> Option<u32> {
        self.ids.get(s).copied()
    }

    pub fn resolve(&self, id: u32) -> &str {
        &self.strings[id as usize]
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }
}
