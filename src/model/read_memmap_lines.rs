use memchr::memchr_iter;
use memmap2::Mmap;
use std::fs::File;

pub struct LineIndex {
    mmap: Mmap,
    starts: Vec<usize>,
}

impl LineIndex {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut starts = vec![0];
        starts.extend(memchr_iter(b'\n', &mmap).map(|i| i + 1));
        // Drop a trailing "start" if the file ends with '\n'
        if starts.last() == Some(&mmap.len()) {
            starts.pop();
        }
        Ok(Self { mmap, starts })
    }

    pub fn line(&self, n: usize) -> Option<&[u8]> {
        let start = *self.starts.get(n)?;
        let end = self
            .starts
            .get(n + 1)
            .map(|&next| next - 1) // exclude the '\n'
            .unwrap_or(self.mmap.len());
        let mut line = &self.mmap[start..end];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        Some(line)
    }
}

fn main() -> std::io::Result<()> {
    let idx = LineIndex::new("data.txt")?;
    println!("{}", String::from_utf8_lossy(idx.line(41).unwrap()));
    Ok(())
}
