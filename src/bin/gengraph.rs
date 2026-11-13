//! Graph-walk generation — no retraining needed.
//!
//! Builds a directed word→word graph at load time from the model's position-0
//! followers (immediate successors) plus a decayed contribution from higher
//! positions. Generation is a biased random walk: from the current word, pick
//! the next word from the graph edges weighted by edge strength, with an
//! additional bias toward words that are also reachable from recent context
//! words (graph proximity instead of positional decay).
//!
//! usage: gengraph <model> <tokens_per_pass> <passes> <eval_jsonl> <eval_tail_lines> <out_tsv> <mode>...
//!   mode = walk[:decay=D]   — pure graph walk from current word with edge decay
//!          ctx[:beta=B]     — blend graph walk with context-node proximity (beta = weight of context bias)
//!          sample           — baseline positional sampling for comparison

use std::collections::VecDeque;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::time::Instant;

use ai2::{Mask, Model, NUMBER_OF_POSITIONS, Token, tokenize};
use hashbrown::{HashMap, HashSet};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const PROMPTS: [&str; 3] = [
    "hello, how are you doing today?",
    "He opened the door and saw",
    "The old man looked at the sea and",
];

#[derive(serde::Deserialize)]
struct TextLine {
    text: String,
}

struct EvalLM {
    vocab: HashMap<String, u32>,
    uni: HashMap<u32, u64>,
    bi: HashMap<(u32, u32), u64>,
    n_tokens: u64,
    unk: u32,
}

impl EvalLM {
    fn id(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.vocab.get(word) { return id; }
        let id = self.vocab.len() as u32;
        self.vocab.insert(word.to_string(), id);
        id
    }
    fn lookup(&self, word: &str) -> u32 {
        self.vocab.get(word).copied().unwrap_or(self.unk)
    }
    fn prob(&self, a: u32, b: u32) -> f64 {
        let n = self.n_tokens as f64;
        let c1 = *self.uni.get(&a).unwrap_or(&0) as f64;
        let c2 = *self.uni.get(&b).unwrap_or(&0) as f64;
        let c12 = *self.bi.get(&(a, b)).unwrap_or(&0) as f64;
        let p = if c12 > 0.0 && c1 > 0.0 {
            0.9 * c12 / c1 + 0.1 * c2 / n
        } else {
            0.05 * c2 / n
        };
        p.max(1e-12)
    }
}

fn rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn fnv(s: &str, salt: u64) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ salt;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn build_eval_lm(path: &str, tail: usize) -> io::Result<(EvalLM, Vec<String>)> {
    let reader = BufReader::new(File::open(path)?);
    let mut ring: VecDeque<String> = VecDeque::with_capacity(tail + 1);
    for line in reader.lines() {
        let line = line?;
        ring.push_back(line);
        if ring.len() > tail { ring.pop_front(); }
    }
    let mut lm = EvalLM {
        vocab: HashMap::new(), uni: HashMap::new(), bi: HashMap::new(),
        n_tokens: 0, unk: 0,
    };
    lm.unk = lm.id("<unk>");
    let lines: Vec<String> = ring.into_iter().collect();
    let mut prev: Option<u32> = None;
    for line in &lines {
        let Ok(t) = serde_json::from_str::<TextLine>(line) else { continue };
        for tok in tokenize(&t.text).tokens() {
            if tok.mask == Mask::Whitespace { continue; }
            let id = lm.id(&tok.content);
            *lm.uni.entry(id).or_insert(0) += 1;
            if let Some(p) = prev { *lm.bi.entry((p, id)).or_insert(0) += 1; }
            prev = Some(id);
            lm.n_tokens += 1;
        }
        prev = None;
    }
    Ok((lm, lines))
}

/// Build graph: word -> list of (successor, weight) from all positions.
/// weight = sum over positions of count * 0.5^(pos/decay)
struct Graph {
    edges: HashMap<String, Vec<(String, f32)>>,
}

fn build_graph(model: &Model, decay: f32) -> Graph {
    let mut edges: HashMap<String, HashMap<String, f32>> = HashMap::new();
    // iterate over all words and their followers at all positions
    for (word, &word_id) in &model.words {
        for pos in 0..NUMBER_OF_POSITIONS as u16 {
            let w = 0.5f32.powf(pos as f32 / decay);
            if w < 1e-4 { break; } // negligible beyond this
            let bucket_id = (word_id as usize) * NUMBER_OF_POSITIONS + pos as usize;
            if let Some(bytes) = model.buckets.record(bucket_id) {
                for chunk in bytes.chunks_exact(8) {
                    let succ_id = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
                    let count = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    let succ = model.strings[succ_id as usize].clone();
                    *edges.entry(word.clone()).or_default().entry(succ).or_insert(0.0) += count as f32 * w;
                }
            }
        }
    }
    let edges = edges
        .into_iter()
        .map(|(k, v)| {
            let mut list: Vec<(String, f32)> = v.into_iter().collect();
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            list.truncate(200); // keep top 200 successors per word
            (k, list)
        })
        .collect();
    Graph { edges }
}

fn pick_from_edges(
    edges: &[(String, f32)],
    rng: &mut StdRng,
) -> String {
    let total: f64 = edges.iter().map(|(_, w)| *w as f64).sum();
    if total <= 0.0 || edges.is_empty() {
        return "NO_WORD_FOUND".to_string();
    }
    let mut x = rng.random::<f64>() * total;
    for (w, wt) in edges {
        x -= *wt as f64;
        if x <= 0.0 {
            return w.clone();
        }
    }
    edges.last().unwrap().0.clone()
}

fn generate_pass(
    model: &mut Model,
    graph: &Graph,
    mode: &str,
    prompt: &str,
    n: usize,
    seed: u64,
) -> (Vec<Token>, usize, f64) {
    let (kind, params) = mode.split_once(':').unwrap_or((mode, ""));
    let mut p: HashMap<&str, f32> = HashMap::new();
    for kv in params.split(',') {
        if let Some((k, v)) = kv.split_once('=') {
            if let Ok(v) = v.parse::<f32>() { p.insert(k, v); }
        }
    }
    let get = |k: &str, default: f32| p.get(k).copied().unwrap_or(default);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut tok = tokenize(prompt);
    let prompt_len = tok.tokens().len();

    let start = Instant::now();
    for _ in 0..n {
        let word = match kind {
            "sample" => {
                // baseline: positional sampling
                let mut scores: HashMap<&str, f32> = HashMap::new();
                for (i, t) in tok.tokens().iter().rev().take(NUMBER_OF_POSITIONS).enumerate() {
                    for (w, wt) in model.followers(&t.content, i as u16) {
                        *scores.entry(w).or_insert(0.0) += wt;
                    }
                }
                let total: f64 = scores.values().map(|&v| v as f64).sum();
                if total > 0.0 {
                    let mut x = rng.random::<f64>() * total;
                    let mut chosen = None;
                    for (w, &wt) in scores.iter() {
                        x -= wt as f64;
                        if x <= 0.0 { chosen = Some(w.to_string()); break; }
                    }
                    chosen.unwrap_or_else(|| scores.keys().next().unwrap().to_string())
                } else {
                    "NO_WORD_FOUND".to_string()
                }
            }
            "walk" | "ctx" => {
                let beta = if kind == "ctx" { get("beta", 0.5) } else { 0.0 };
                // current word = last non-whitespace token
                let last_word = tok
                    .tokens()
                    .iter()
                    .rev()
                    .find(|t| t.mask != Mask::Whitespace)
                    .map(|t| t.content.clone())
                    .unwrap_or_default();

                let mut scores: HashMap<String, f64> = HashMap::new();
                // graph edges from current word
                if let Some(edges) = graph.edges.get(&last_word) {
                    for (succ, wt) in edges {
                        *scores.entry(succ.clone()).or_insert(0.0) += *wt as f64;
                    }
                }
                // context bias: for each recent context word, add its graph edges
                if beta > 0.0 {
                    for t in tok.tokens().iter().rev().take(20).skip(1) {
                        if t.mask == Mask::Whitespace { continue; }
                        if let Some(edges) = graph.edges.get(&t.content) {
                            for (succ, wt) in edges {
                                *scores.entry(succ.clone()).or_insert(0.0) += beta as f64 * *wt as f64;
                            }
                        }
                    }
                }
                if scores.is_empty() {
                    // fallback to positional
                    let mut fb: HashMap<&str, f32> = HashMap::new();
                    for (i, t) in tok.tokens().iter().rev().take(NUMBER_OF_POSITIONS).enumerate() {
                        for (w, wt) in model.followers(&t.content, i as u16) {
                            *fb.entry(w).or_insert(0.0) += wt;
                        }
                    }
                    let total: f64 = fb.values().map(|&v| v as f64).sum();
                    if total > 0.0 {
                        let mut x = rng.random::<f64>() * total;
                        let mut chosen = None;
                        for (w, &wt) in fb.iter() {
                            x -= wt as f64;
                            if x <= 0.0 { chosen = Some(w.to_string()); break; }
                        }
                        chosen.unwrap_or_else(|| fb.keys().next().unwrap().to_string())
                    } else {
                        "NO_WORD_FOUND".to_string()
                    }
                } else {
                    let total: f64 = scores.values().sum();
                    let mut x = rng.random::<f64>() * total;
                    let mut chosen = None;
                    for (w, &wt) in scores.iter() {
                        x -= wt;
                        if x <= 0.0 { chosen = Some(w.clone()); break; }
                    }
                    chosen.unwrap_or_else(|| scores.keys().next().unwrap().clone())
                }
            }
            _ => panic!("unknown mode '{kind}'"),
        };

        let token = Token {
            mask: Mask::classify_word(&word),
            content: word,
        };
        tok.push(token.clone());
        if token.mask == Mask::Punctuation {
            tok.push(Token::space());
        }
    }
    let secs = start.elapsed().as_secs_f64();
    (tok.tokens().to_vec(), prompt_len, secs)
}

struct Metrics {
    ppl: f64, repeat4: f64, ttr: f64, pos0: f64, avgsent: f64, oov: f64,
}

fn compute_metrics(model: &Model, lm: &EvalLM, full: &[Token], prompt_len: usize) -> Metrics {
    let nonws: Vec<&Token> = full.iter().filter(|t| t.mask != Mask::Whitespace).collect();
    let gen_start = prompt_len.saturating_sub(1).min(nonws.len());
    let gen_toks = &nonws[gen_start..];

    let ids: Vec<u32> = gen_toks.iter().map(|t| lm.lookup(&t.content)).collect();
    let mut nll = 0.0;
    for w in ids.windows(2) { nll += -lm.prob(w[0], w[1]).ln(); }
    let ppl = if ids.len() > 1 { (nll / (ids.len() - 1) as f64).exp() } else { f64::INFINITY };

    let mut seen: HashSet<(&str, &str, &str, &str)> = HashSet::new();
    let mut total4 = 0usize;
    let mut dup4 = 0usize;
    let contents: Vec<&str> = gen_toks.iter().map(|t| t.content.as_str()).collect();
    for w in contents.windows(4) {
        total4 += 1;
        if !seen.insert((w[0], w[1], w[2], w[3])) { dup4 += 1; }
    }
    let repeat4 = if total4 > 0 { dup4 as f64 / total4 as f64 } else { 0.0 };

    let words: Vec<&&Token> = gen_toks.iter().filter(|t| t.mask == Mask::None).collect();
    let mut uniq: HashSet<&str> = HashSet::new();
    let mut oov = 0usize;
    for t in &words {
        uniq.insert(t.content.as_str());
        if !lm.vocab.contains_key(t.content.as_str()) { oov += 1; }
    }
    let ttr = if words.is_empty() { 0.0 } else { uniq.len() as f64 / words.len() as f64 };
    let oov = if words.is_empty() { 0.0 } else { oov as f64 / words.len() as f64 };

    let mut sentences = 0usize;
    for t in gen_toks {
        if t.mask == Mask::Punctuation && matches!(t.content.as_str(), "." | "!" | "?") {
            sentences += 1;
        }
    }
    let avgsent = if sentences > 0 { words.len() as f64 / sentences as f64 } else { words.len() as f64 };

    let mut valid = 0usize;
    let mut total_pairs = 0usize;
    for w in gen_toks.windows(2) {
        let (a, b) = (&w[0].content, &w[1].content);
        if let (Some(&ra), Some(&rb)) = (model.words.get(a.as_str()), model.words.get(b.as_str())) {
            total_pairs += 1;
            if let Some(bytes) = model.buckets.record(ra as usize * NUMBER_OF_POSITIONS) {
                for c in bytes.chunks_exact(8) {
                    if u32::from_le_bytes(c[0..4].try_into().unwrap()) == rb {
                        valid += 1; break;
                    }
                }
            }
        }
    }
    let pos0 = if total_pairs > 0 { valid as f64 / total_pairs as f64 } else { 0.0 };

    Metrics { ppl, repeat4, ttr, pos0, avgsent, oov }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!(
            "usage: {} <model> <tokens_per_pass> <passes> <eval_jsonl> <eval_tail_lines> <out_tsv> <mode>...",
            args[0]
        );
        std::process::exit(1);
    }
    let model_name = &args[1];
    let tokens_per_pass: usize = args[2].parse().unwrap();
    let passes: usize = args[3].parse().unwrap();
    let eval_jsonl = &args[4];
    let eval_tail: usize = args[5].parse().unwrap();
    let out_path = &args[6];
    let modes = &args[7..];

    let mut out = io::BufWriter::new(File::create(out_path).unwrap());
    writeln!(out, "pass\tmode\tppl\trepeat4\tttr\tpos0\tavgsent\toov\tntok\tsecs\trss_mb").unwrap();

    eprintln!("building eval LM...");
    let (lm, _lines) = build_eval_lm(eval_jsonl, eval_tail).unwrap();
    eprintln!("eval LM: {} tokens, rss {:.0} MB", lm.n_tokens, rss_mb());

    eprintln!("loading model '{model_name}'...");
    let mut model = Model::load(model_name).unwrap();
    eprintln!("model loaded, rss {:.0} MB", rss_mb());

    eprintln!("building graph (decay=2.0)...");
    let t = Instant::now();
    let graph = build_graph(&model, 2.0);
    eprintln!("graph built: {} nodes, {:.1}s, rss {:.0} MB", graph.edges.len(), t.elapsed().as_secs_f64(), rss_mb());

    struct Row { mode: String, m: Metrics, secs: f64 }
    let mut rows: Vec<Row> = Vec::new();

    for mode in modes {
        for pass in 0..passes {
            let prompt = PROMPTS[pass % PROMPTS.len()];
            let seed = fnv(mode, pass as u64);
            let (full, prompt_len, secs) = generate_pass(&mut model, &graph, mode, prompt, tokens_per_pass, seed);
            let m = compute_metrics(&model, &lm, &full, prompt_len);
            let rss = rss_mb();
            writeln!(
                out,
                "pass\t{mode}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{}\t{:.1}\t{:.0}",
                m.ppl, m.repeat4, m.ttr, m.pos0, m.avgsent, m.oov,
                full.len() - prompt_len, secs, rss
            ).unwrap();
            out.flush().unwrap();
            rows.push(Row { mode: mode.clone(), m, secs });
            if rss > 16_000.0 {
                eprintln!("WARNING: rss {:.0} MB over 16 GB, aborting", rss);
                std::process::exit(2);
            }
        }
    }

    writeln!(out, "sum\tmode\tppl_mean\tppl_std\trepeat4_mean\tpos0_mean\tttr_mean\tavgsent_mean\toov_mean\tsecs_total").unwrap();
    for mode in modes {
        let rs: Vec<&Row> = rows.iter().filter(|r| r.mode == *mode).collect();
        if rs.is_empty() { continue; }
        let col = |f: fn(&Metrics) -> f64| rs.iter().map(|r| f(&r.m)).collect::<Vec<_>>();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let std = |v: &[f64]| {
            let m = mean(v);
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let secs_total: f64 = rs.iter().map(|r| r.secs).sum();
        writeln!(
            out,
            "sum\t{mode}\t{:.2}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{:.1}",
            mean(&col(|m| m.ppl)), std(&col(|m| m.ppl)), mean(&col(|m| m.repeat4)),
            mean(&col(|m| m.pos0)), mean(&col(|m| m.ttr)), mean(&col(|m| m.avgsent)),
            mean(&col(|m| m.oov)), secs_total
        ).unwrap();
    }
    out.flush().unwrap();
    eprintln!("done. rss {:.0} MB", rss_mb());
}
