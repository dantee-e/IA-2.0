use memmap2::Mmap;
use std::{fs::File, path::PathBuf};

pub struct LineIndex {
    mmap: Mmap,
    pub starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(path: PathBuf, starts: Vec<usize>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self { mmap, starts })
    }

    pub fn record(&self, n: usize) -> Option<&[u8]> {
        let start = *self.starts.get(n)?;
        let end = *self.starts.get(n + 1)?; // sentinel guarantees this exists for every real record
        self.mmap.get(start..end)
    }
}
