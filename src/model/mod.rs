use hashbrown::HashMap;
pub mod persist;

pub type Model = HashMap<String, HashMap<usize, HashMap<String, usize>>>;
