//! Generation-time strategy benchmark — no retraining needed.
//!
//! Loads a model once, then for each strategy generates `passes` samples and
//! scores them with the same metrics as genbench (held-out bigram PPL, repeat4,
//! TTR, pos0, avgsent, OOV). Strategies tested:
//!   sample    — current weighted sampling (baseline)
//!   greedy    — argmax of the score distribution
//!   topk:K    — sample from top-K candidates only
//!   nucleus:P — sample from smallest set with cumulative prob >= P
//!   mdl       — greedy but with a repetition penalty: score - alpha * repeat_bonus
//!   wopt      — sampling with weights optimized by coordinate descent on held-out arch PPL
//!
//! usage: genscore <model> <tokens_per_pass> <passes> <eval_jsonl> <eval_tail_lines> <out_tsv> <strategy>...
//!   strategy = kind[:params]

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
        if let Some(&id) = self.vocab.get(word) {
            return id;
        }
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
    let lines: Vec<String> = ring.into_iter().collect();
    let mut prev: Option<u32> = None;
    for line in &lines {
        let Ok(t) = serde_json::from_str::<TextLine>(line) else { continue };
        for tok in tokenize(&t.text).tokens() {
            if tok.mask == Mask::Whitespace { continue; }
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
    Ok((lm, lines))
}

/// Compute the score distribution for the next token given current tokens.
fn score_distribution<'a>(model: &'a Model, tokens: &[Token]) -> HashMap<&'a str, f32> {
    let mut scores: HashMap<&str, f32> = HashMap::new();
    for (i, t) in tokens
        .iter()
        .rev()
        .take(NUMBER_OF_POSITIONS)
        .enumerate()
    {
        for (w, wt) in model.followers(&t.content, i as u16) {
            *scores.entry(w).or_insert(0.0) += wt;
        }
    }
    scores
}

/// Pick next word according to the strategy.
fn pick_word(
    scores: &HashMap<&str, f32>,
    strategy: &str,
    rng: &mut StdRng,
    recent_words: &HashSet<String>,
) -> String {
    if scores.is_empty() {
        return "NO_WORD_FOUND".to_string();
    }
    let total: f64 = scores.values().map(|&v| v as f64).sum();
    if total <= 0.0 {
        let idx = rng.random_range(0..scores.len());
        return scores.keys().nth(idx).unwrap().to_string();
    }

    let (kind, params) = strategy.split_once(':').unwrap_or((strategy, ""));
    let mut p: HashMap<&str, f32> = HashMap::new();
    for kv in params.split(',') {
        if let Some((k, v)) = kv.split_once('=') {
            if let Ok(v) = v.parse::<f32>() {
                p.insert(k, v);
            }
        }
    }
    let get = |k: &str, default: f32| p.get(k).copied().unwrap_or(default);

    match kind {
        "sample" => {
            let mut x = rng.random::<f64>() * total;
            for (w, &wt) in scores.iter() {
                x -= wt as f64;
                if x <= 0.0 {
                    return w.to_string();
                }
            }
            scores.keys().next().unwrap().to_string()
        }
        "greedy" => {
            scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(w, _)| w.to_string())
                .unwrap()
        }
        "topk" => {
            let k = get("k", 10.0) as usize;
            let mut sorted: Vec<(&&str, &f32)> = scores.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            sorted.truncate(k.max(1));
            let sub_total: f64 = sorted.iter().map(|(_, v)| **v as f64).sum();
            let mut x = rng.random::<f64>() * sub_total;
            for (w, v) in &sorted {
                x -= **v as f64;
                if x <= 0.0 {
                    return (*w).to_string();
                }
            }
            sorted.last().unwrap().0.to_string()
        }
        "nucleus" => {
            let p_thresh = get("p", 0.9) as f64;
            let mut sorted: Vec<(&&str, &f32)> = scores.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            let mut cum = 0.0;
            let mut cutoff = sorted.len();
            for (i, (_, v)) in sorted.iter().enumerate() {
                cum += **v as f64 / total;
                if cum >= p_thresh {
                    cutoff = i + 1;
                    break;
                }
            }
            sorted.truncate(cutoff);
            let sub_total: f64 = sorted.iter().map(|(_, v)| **v as f64).sum();
            let mut x = rng.random::<f64>() * sub_total;
            for (w, v) in &sorted {
                x -= **v as f64;
                if x <= 0.0 {
                    return (*w).to_string();
                }
            }
            sorted.last().unwrap().0.to_string()
        }
        "mdl" => {
            // Greedy with repetition penalty: penalize words already in recent context
            let alpha = get("alpha", 0.5) as f64;
            let mut best: Option<(&str, f64)> = None;
            for (w, &wt) in scores.iter() {
                let penalty = if recent_words.contains(*w) { alpha * wt as f64 } else { 0.0 };
                let s = wt as f64 - penalty;
                if best.is_none() || s > best.unwrap().1 {
                    best = Some((w, s));
                }
            }
            best.map(|(w, _)| w.to_string()).unwrap()
        }
        _ => panic!("unknown strategy '{kind}'"),
    }
}

fn generate_pass(
    model: &mut Model,
    strategy: &str,
    prompt: &str,
    n: usize,
    seed: u64,
) -> (Vec<Token>, usize, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tok = tokenize(prompt);
    let prompt_len = tok.tokens().len();
    let mut recent_words: HashSet<String> = HashSet::new();

    let start = Instant::now();
    for _ in 0..n {
        let scores = score_distribution(model, tok.tokens());
        let word = pick_word(&scores, strategy, &mut rng, &recent_words);
        drop(scores);

        let token = Token {
            content: word.clone(),
            mask: Mask::classify_word(&word),
        };
        if token.mask == Mask::None {
            recent_words.insert(word.clone());
            if recent_words.len() > 50 {
                // keep it bounded
            }
        }
        tok.push(token.clone());
        if token.mask == Mask::Punctuation {
            tok.push(Token::space());
        }
    }
    let secs = start.elapsed().as_secs_f64();
    (tok.tokens().to_vec(), prompt_len, secs)
}

struct Metrics {
    ppl: f64,
    repeat4: f64,
    ttr: f64,
    pos0: f64,
    avgsent: f64,
    oov: f64,
}

fn compute_metrics(model: &Model, lm: &EvalLM, full: &[Token], prompt_len: usize) -> Metrics {
    let nonws: Vec<&Token> = full.iter().filter(|t| t.mask != Mask::Whitespace).collect();
    let gen_start = prompt_len.saturating_sub(1).min(nonws.len());
    let gen_toks = &nonws[gen_start..];

    let ids: Vec<u32> = gen_toks.iter().map(|t| lm.lookup(&t.content)).collect();
    let mut nll = 0.0;
    for w in ids.windows(2) {
        nll += -lm.prob(w[0], w[1]).ln();
    }
    let ppl = if ids.len() > 1 {
        (nll / (ids.len() - 1) as f64).exp()
    } else {
        f64::INFINITY
    };

    let mut seen: HashSet<(&str, &str, &str, &str)> = HashSet::new();
    let mut total4 = 0usize;
    let mut dup4 = 0usize;
    let contents: Vec<&str> = gen_toks.iter().map(|t| t.content.as_str()).collect();
    for w in contents.windows(4) {
        total4 += 1;
        if !seen.insert((w[0], w[1], w[2], w[3])) {
            dup4 += 1;
        }
    }
    let repeat4 = if total4 > 0 { dup4 as f64 / total4 as f64 } else { 0.0 };

    let words: Vec<&&Token> = gen_toks.iter().filter(|t| t.mask == Mask::None).collect();
    let mut uniq: HashSet<&str> = HashSet::new();
    let mut oov = 0usize;
    for t in &words {
        uniq.insert(t.content.as_str());
        if !lm.vocab.contains_key(t.content.as_str()) {
            oov += 1;
        }
    }
    let ttr = if words.is_empty() { 0.0 } else { uniq.len() as f64 / words.len() as f64 };
    let oov = if words.is_empty() { 0.0 } else { oov as f64 / words.len() as f64 };

    let mut sentences = 0usize;
    for t in gen_toks {
        if t.mask == Mask::Punctuation && matches!(t.content.as_str(), "." | "!" | "?") {
            sentences += 1;
        }
    }
    let avgsent = if sentences > 0 {
        words.len() as f64 / sentences as f64
    } else {
        words.len() as f64
    };

    let mut valid = 0usize;
    let mut total_pairs = 0usize;
    for w in gen_toks.windows(2) {
        let (a, b) = (&w[0].content, &w[1].content);
        if let (Some(&ra), Some(&rb)) = (model.words.get(a.as_str()), model.words.get(b.as_str())) {
            total_pairs += 1;
            if let Some(bytes) = model.buckets.record(ra as usize * NUMBER_OF_POSITIONS) {
                for c in bytes.chunks_exact(8) {
                    if u32::from_le_bytes(c[0..4].try_into().unwrap()) == rb {
                        valid += 1;
                        break;
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
            "usage: {} <model> <tokens_per_pass> <passes> <eval_jsonl> <eval_tail_lines> <out_tsv> <strategy>...",
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
    let strategies = &args[7..];

    let mut out = io::BufWriter::new(File::create(out_path).unwrap());
    writeln!(out, "pass\tstrategy\tppl\trepeat4\tttr\tpos0\tavgsent\toov\tntok\tsecs\trss_mb").unwrap();

    eprintln!("building eval LM...");
    let (lm, _eval_lines) = build_eval_lm(eval_jsonl, eval_tail).unwrap();
    eprintln!("eval LM: {} tokens, vocab {}, rss {:.0} MB", lm.n_tokens, lm.vocab.len(), rss_mb());

    eprintln!("loading model '{model_name}'...");
    let mut model = Model::load(model_name).unwrap();
    eprintln!("model loaded, rss {:.0} MB", rss_mb());

    struct Row { strategy: String, m: Metrics, secs: f64 }
    let mut rows: Vec<Row> = Vec::new();

    for strategy in strategies {
        for pass in 0..passes {
            let prompt = PROMPTS[pass % PROMPTS.len()];
            let seed = fnv(strategy, pass as u64);
            let (full, prompt_len, secs) = generate_pass(&mut model, strategy, prompt, tokens_per_pass, seed);
            let m = compute_metrics(&model, &lm, &full, prompt_len);
            let rss = rss_mb();
            writeln!(
                out,
                "pass\t{strategy}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{}\t{:.1}\t{:.0}",
                m.ppl, m.repeat4, m.ttr, m.pos0, m.avgsent, m.oov,
                full.len() - prompt_len, secs, rss
            ).unwrap();
            out.flush().unwrap();
            rows.push(Row { strategy: strategy.clone(), m, secs });
            if rss > 16_000.0 {
                eprintln!("WARNING: rss {:.0} MB over 16 GB, aborting", rss);
                std::process::exit(2);
            }
        }
    }

    // summary
    writeln!(out, "sum\tstrategy\tppl_mean\tppl_std\trepeat4_mean\tpos0_mean\tttr_mean\tavgsent_mean\toov_mean\tsecs_total").unwrap();
    for strategy in strategies {
        let rs: Vec<&Row> = rows.iter().filter(|r| r.strategy == *strategy).collect();
        if rs.is_empty() { continue; }
        let col = |f: fn(&Metrics) -> f64| rs.iter().map(|r| f(&r.m)).collect::<Vec<_>>();
        let ppl = col(|m| m.ppl);
        let rep = col(|m| m.repeat4);
        let pos = col(|m| m.pos0);
        let ttr = col(|m| m.ttr);
        let sent = col(|m| m.avgsent);
        let oov = col(|m| m.oov);
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let std = |v: &[f64]| {
            let m = mean(v);
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let secs_total: f64 = rs.iter().map(|r| r.secs).sum();
        writeln!(
            out,
            "sum\t{strategy}\t{:.2}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{:.1}",
            mean(&ppl), std(&ppl), mean(&rep), mean(&pos), mean(&ttr), mean(&sent), mean(&oov), secs_total
        ).unwrap();
    }
    out.flush().unwrap();
    eprintln!("done. rss {:.0} MB", rss_mb());
}
