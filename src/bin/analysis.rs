//! Dev analysis tools for the generator model.
//!
//! subcommands:
//!   sparsity <model>
//!       Streams the model's buckets.bin once and reports count histograms,
//!       overall and per position (are far positions mostly singletons?).
//!   ceiling <model> <eval_jsonl> <eval_tail> <start_line> <n_lines> <tokens_per_line> <spec...>
//!       Scores REAL held-out text (lines start_line..start_line+n_lines of the
//!       jsonl, training never saw them) under the model's own position-mixture
//!       distribution, for each weight variant spec. Reports:
//!         ppl_arch  - perplexity of real text under the mixture (floored at 1e-9)
//!         zero_rate - fraction of real tokens the mixture assigns ~0 probability
//!         ppl_bigram- perplexity of the same tokens under a held-out bigram LM
//!       (the bigram LM is built from the last eval_tail lines, same as genbench)
//!
//! variant spec kinds: same as genbench plus `uniform` (all weights 1.0).

use std::collections::VecDeque;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

use ai2::{NUMBER_OF_POSITIONS, Model, tokenize};
use hashbrown::{HashMap, HashSet};

#[derive(serde::Deserialize)]
struct TextLine {
    text: String,
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

fn parse_variant(spec: &str) -> [f32; NUMBER_OF_POSITIONS] {
    let (kind, params) = spec.split_once(':').unwrap_or((spec, ""));
    let mut p: HashMap<&str, f32> = HashMap::new();
    for kv in params.split(',') {
        if let Some((k, v)) = kv.split_once('=') {
            if let Ok(v) = v.parse::<f32>() {
                p.insert(k, v);
            }
        }
    }
    let get = |k: &str, default: f32| p.get(k).copied().unwrap_or(default);

    std::array::from_fn(|i| {
        let pos = i as f32;
        let v = match kind {
            "half" => 0.5f32.powf(pos / get("hl", 2.0)),
            "linear" => 1.0 - pos / get("range", 300.0),
            "osc" => {
                0.5f32.powf(pos / get("hl", 2.0))
                    * (1.0
                        + get("amp", 0.5)
                            * (std::f32::consts::TAU * pos / get("period", 8.0)).cos())
            }
            "inv" => 1.0 / (1.0 + pos / get("k", 8.0)),
            "power" => (1.0 - pos / NUMBER_OF_POSITIONS as f32).powf(get("p", 1.0)),
            "gauss" => (-0.5 * (pos / get("s", 16.0)).powi(2)).exp(),
            "step" => {
                if pos < get("k", 16.0) {
                    1.0
                } else {
                    0.0
                }
            }
            "tukey" => {
                0.5 * (1.0 + (std::f32::consts::PI * pos / NUMBER_OF_POSITIONS as f32).cos())
            }
            "log" => 1.0 - (1.0 + pos).ln() / (1.0 + NUMBER_OF_POSITIONS as f32).ln(),
            "saw" => {
                let period = get("period", 8.0);
                let ph = (pos % period) / period;
                0.5f32.powf(pos / get("hl", 8.0)) * (1.0 - (2.0 * ph - 1.0).abs())
            }
            "square" => {
                let c = (std::f32::consts::TAU * pos / get("period", 8.0)).cos();
                let amp = get("amp", 0.5);
                0.5f32.powf(pos / get("hl", 2.0)) * if c >= 0.0 { 1.0 + amp } else { 1.0 - amp }
            }
            "logper" => {
                0.5f32.powf(pos / get("hl", 2.0))
                    * (1.0
                        + get("amp", 0.5)
                            * (std::f32::consts::TAU * (1.0 + pos).log2() / get("period", 16.0).log2())
                                .cos())
            }
            "piecewise" => {
                let k = get("k", 16.0);
                if pos < k {
                    1.0
                } else {
                    0.5f32.powf((pos - k) / get("hl", 4.0))
                }
            }
            "uniform" => 1.0,
            other => panic!("unknown variant kind '{other}'"),
        };
        v.max(0.0)
    })
}

fn sparsity(model_name: &str) {
    let model = Model::load(model_name).unwrap();
    let n_buckets = model.strings.len() * NUMBER_OF_POSITIONS;
    eprintln!("model '{model_name}': {} words, {n_buckets} buckets, rss {:.0} MB", model.strings.len(), rss_mb());

    let mut total_pairs: u64 = 0;
    let mut total_count: u64 = 0;
    let mut singletons: u64 = 0;
    let mut le3: u64 = 0;
    let mut ge100: u64 = 0;
    // per-position singleton stats: (pairs, singletons) for pos 0..9 and deciles
    let mut pos_pairs = [0u64; NUMBER_OF_POSITIONS];
    let mut pos_single = [0u64; NUMBER_OF_POSITIONS];

    for id in 0..n_buckets {
        let Some(bytes) = model.buckets.record(id) else {
            continue;
        };
        let pos = (id % NUMBER_OF_POSITIONS) as usize;
        for c in bytes.chunks_exact(8) {
            let count = u32::from_le_bytes(c[4..8].try_into().unwrap()) as u64;
            total_pairs += 1;
            total_count += count;
            pos_pairs[pos] += 1;
            if count == 1 {
                singletons += 1;
                pos_single[pos] += 1;
            }
            if count <= 3 {
                le3 += 1;
            }
            if count >= 100 {
                ge100 += 1;
            }
        }
        if id % 10_000_000 == 0 {
            eprintln!("  ... {id}/{n_buckets} buckets, rss {:.0} MB", rss_mb());
        }
    }

    println!("total pairs: {total_pairs}, total count: {total_count}, avg count {:.2}", total_count as f64 / total_pairs as f64);
    println!("singletons (count==1): {singletons} ({:.1}%)", 100.0 * singletons as f64 / total_pairs as f64);
    println!("count<=3: {le3} ({:.1}%)", 100.0 * le3 as f64 / total_pairs as f64);
    println!("count>=100: {ge100} ({:.2}%)", 100.0 * ge100 as f64 / total_pairs as f64);
    println!("pos\tpairs\tsingleton%");
    for pos in 0..NUMBER_OF_POSITIONS {
        if pos < 10 || pos % 25 == 0 || pos == NUMBER_OF_POSITIONS - 1 {
            let p = pos_pairs[pos] as f64;
            println!(
                "{pos}\t{}\t{:.1}",
                pos_pairs[pos],
                if p > 0.0 { 100.0 * pos_single[pos] as f64 / p } else { 0.0 }
            );
        }
    }
}

struct EvalLM {
    vocab: HashMap<String, u32>,
    uni: HashMap<u32, u64>,
    bi: HashMap<(u32, u32), u64>,
    n_tokens: u64,
    unk: u32,
}

impl EvalLM {
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

fn build_eval_lm(path: &str, tail: usize) -> io::Result<EvalLM> {
    let reader = BufReader::new(File::open(path)?);
    let mut ring: VecDeque<String> = VecDeque::with_capacity(tail + 1);
    for line in reader.lines() {
        let line = line?;
        ring.push_back(line);
        if ring.len() > tail {
            ring.pop_front();
        }
    }
    let mut lm = EvalLM {
        vocab: HashMap::new(),
        uni: HashMap::new(),
        bi: HashMap::new(),
        n_tokens: 0,
        unk: 0,
    };
    lm.unk = lm.id("<unk>");
    let mut prev: Option<u32> = None;
    for line in &ring {
        let Ok(t) = serde_json::from_str::<TextLine>(line) else {
            continue;
        };
        for tok in tokenize(&t.text).tokens() {
            let id = lm.id(&tok.content);
            *lm.uni.entry(id).or_insert(0) += 1;
            if let Some(p) = prev {
                *lm.bi.entry((p, id)).or_insert(0) += 1;
            }
            prev = Some(id);
            lm.n_tokens += 1;
        }
        prev = None;
    }
    Ok(lm)
}

impl EvalLM {
    fn id(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.vocab.get(word) {
            return id;
        }
        let id = self.vocab.len() as u32;
        self.vocab.insert(word.to_string(), id);
        id
    }
}

/// Reads lines [start_line, start_line+n_lines) (0-based) from the jsonl.
fn read_lines(path: &str, start_line: usize, n_lines: usize) -> io::Result<Vec<String>> {
    let reader = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        if i < start_line {
            continue;
        }
        if i >= start_line + n_lines {
            break;
        }
        let line = line?;
        if let Ok(t) = serde_json::from_str::<TextLine>(&line) {
            out.push(t.text);
        }
    }
    Ok(out)
}

fn ceiling(
    model_name: &str,
    eval_jsonl: &str,
    eval_tail: usize,
    start_line: usize,
    n_lines: usize,
    tokens_per_line: usize,
    specs: &[String],
) {
    eprintln!("building eval bigram LM (tail {eval_tail})...");
    let lm = build_eval_lm(eval_jsonl, eval_tail).unwrap();
    eprintln!("eval LM: {} tokens, {} bigrams, rss {:.0} MB", lm.n_tokens, lm.bi.len(), rss_mb());

    eprintln!("loading model '{model_name}'...");
    let mut model = Model::load(model_name).unwrap();
    eprintln!("model loaded, rss {:.0} MB", rss_mb());

    let texts = read_lines(eval_jsonl, start_line, n_lines).unwrap();
    eprintln!("scoring {} held-out lines (from line {start_line}), up to {tokens_per_line} tokens each", texts.len());

    // Precompute the eval-LM bigram PPL of the selected tokens once.
    let mut docs: Vec<Vec<String>> = Vec::new();
    let mut bi_nll = 0.0;
    let mut bi_n = 0usize;
    for text in &texts {
        let toks: Vec<String> = tokenize(text)
            .tokens()
            .iter()
            .take(tokens_per_line)
            .map(|t| t.content.clone())
            .collect();
        let ids: Vec<u32> = toks.iter().map(|t| lm.lookup(t)).collect();
        for w in ids.windows(2) {
            bi_nll += -lm.prob(w[0], w[1]).ln();
            bi_n += 1;
        }
        docs.push(toks);
    }
    let ppl_bigram = (bi_nll / bi_n as f64).exp();
    println!("tokens scored: {bi_n}  ppl_bigram(real text) = {ppl_bigram:.2}");

    println!("variant\tppl_arch\tzero_rate\tsecs\trss_mb");
    for spec in specs {
        let weights = parse_variant(spec);
        model.weights = weights;
        let start = std::time::Instant::now();
        let mut nll = 0.0;
        let mut n = 0usize;
        let mut zeros = 0usize;
        for toks in &docs {
            // context starts empty; each token is scored given up to 300 previous
            let mut ctx: Vec<&str> = Vec::new();
            for w in toks {
                if !ctx.is_empty() {
                    let mut scores: HashMap<&str, f32> = HashMap::new();
                    for (i, c) in ctx.iter().rev().take(NUMBER_OF_POSITIONS).enumerate() {
                        for (f, wt) in model.followers(c, i as u16) {
                            *scores.entry(f).or_insert(0.0) += wt;
                        }
                    }
                    let total: f64 = scores.values().map(|&v| v as f64).sum();
                    let s = scores.get(w.as_str()).copied().unwrap_or(0.0) as f64;
                    let p = if total > 0.0 { s / total } else { 0.0 };
                    if p <= 1e-9 {
                        zeros += 1;
                        nll += -1e-9f64.ln();
                    } else {
                        nll += -p.ln();
                    }
                    n += 1;
                }
                ctx.push(w.as_str());
            }
        }
        let ppl_arch = (nll / n as f64).exp();
        println!(
            "{spec}\t{ppl_arch:.2}\t{:.4}\t{:.1}\t{:.0}",
            zeros as f64 / n as f64,
            start.elapsed().as_secs_f64(),
            rss_mb()
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("sparsity") if args.len() == 3 => sparsity(&args[2]),
        Some("ceiling") if args.len() >= 9 => ceiling(
            &args[2],
            &args[3],
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
            args[6].parse().unwrap(),
            args[7].parse().unwrap(),
            &args[8..],
        ),
        _ => {
            eprintln!("usage: analysis sparsity <model>\n       analysis ceiling <model> <eval_jsonl> <eval_tail> <start_line> <n_lines> <tokens_per_line> <spec...>");
            std::process::exit(1);
        }
    }
}
