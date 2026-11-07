use std::collections::HashMap;

/// Maps each distinct token string to a u32 id, and back.
#[derive(Default)]
pub struct Interner {
    ids: HashMap<String, u32>,
    strings: Vec<String>,
}

impl Interner {}
