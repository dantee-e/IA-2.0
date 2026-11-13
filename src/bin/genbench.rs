//! Benchmark for position-weight functions in the generator.
//!
//! Loads a generation model once, then for each weight-variant spec generates
//! `passes` samples of `tokens` tokens (one fixed prompt per pass, seeded by
//! variant+pass) and scores them with:
//!   ppl     - perplexity under a bigram+unigram-backoff LM built from HELD-OUT
//!             corpus lines (tail of the training jsonl that training never saw)
//!   repeat4 - fraction of 4-grams (over non-whitespace tokens) already seen in
//!             the same sample (loopiness; lower is better)
//!   ttr     - type-token ratio over word tokens (diversity)
//!   pos0    - fraction of adjacent generated token pairs that exist in the
//!             model at position 0, i.e. word b really follows word a in the
//!             corpus (local grammatical soundness; higher is better)
//!   avgsent - average words per sentence (terminal . ! ?)
//!   oov     - fraction of generated word tokens missing from the eval vocab
//!
//! usage: genbench <model> <tokens_per_pass> <passes> <eval_jsonl> <eval_tail_lines> <out_prefix> <variant_spec>...
//!   variant_spec = kind:k=v,k=v   (kinds: half linear osc inv power gauss step
//!                                  tukey log saw square logper piecewise)
//!
//! Output: TSV on stdout — `pass` rows per run, `sum` rows per variant, `rank`
//! rows at the end sorted by mean ppl. Generated pass-0 texts are written to
//! `<out_prefix>_<variant>.txt`. VmRSS is logged on every row.

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

    /// bigram-with-unigram-backoff probability of `b` after `a`
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
        let Ok(t) = serde_json::from_str::<TextLine>(line) else {
            continue;
        };
        for tok in tokenize(&t.text).tokens() {
            if tok.mask == Mask::Whitespace {
                continue;
            }
            let id = lm.id(&tok.content);
            *lm.uni.entry(id).or_insert(0) += 1;
            if let Some(p) = prev {
                *lm.bi.entry((p, id)).or_insert(0) += 1;
            }
            prev = Some(id);
            lm.n_tokens += 1;
        }
        // sentence boundary between lines
        prev = None;
    }
    Ok((lm, lines))
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
            "half" => {
                let hl = get("hl", 2.0);
                0.5f32.powf(pos / hl)
            }
            "linear" => {
                let range = get("range", 300.0);
                1.0 - pos / range
            }
            "osc" => {
                let hl = get("hl", 2.0);
                let period = get("period", 8.0);
                let amp = get("amp", 0.5);
                0.5f32.powf(pos / hl) * (1.0 + amp * (std::f32::consts::TAU * pos / period).cos())
            }
            "inv" => {
                let k = get("k", 8.0);
                1.0 / (1.0 + pos / k)
            }
            "power" => {
                let e = get("p", 1.0);
                (1.0 - pos / NUMBER_OF_POSITIONS as f32).powf(e)
            }
            "gauss" => {
                let s = get("s", 16.0);
                (-0.5 * (pos / s) * (pos / s)).exp()
            }
            "step" => {
                let k = get("k", 16.0);
                if pos < k { 1.0 } else { 0.0 }
            }
            "tukey" => 0.5 * (1.0 + (std::f32::consts::PI * pos / NUMBER_OF_POSITIONS as f32).cos()),
            "log" => 1.0 - (1.0 + pos).ln() / (1.0 + NUMBER_OF_POSITIONS as f32).ln(),
            "saw" => {
                let period = get("period", 8.0);
                let hl = get("hl", 8.0);
                let ph = (pos % period) / period;
                let tri = 1.0 - (2.0 * ph - 1.0).abs();
                0.5f32.powf(pos / hl) * (0.5 + 0.5 * tri)
            }
            "square" => {
                let hl = get("hl", 2.0);
                let period = get("period", 8.0);
                let amp = get("amp", 0.5);
                let c = (std::f32::consts::TAU * pos / period).cos();
                let s = if c >= 0.0 { 1.0 + amp } else { (1.0 - amp).max(0.0) };
                0.5f32.powf(pos / hl) * s
            }
            "logper" => {
                let hl = get("hl", 2.0);
                let period = get("period", 16.0);
                let amp = get("amp", 0.5);
                0.5f32.powf(pos / hl)
                    * (1.0 + amp * (std::f32::consts::TAU * (1.0 + pos).log2() / period.log2()).cos())
            }
            "piecewise" => {
                let k = get("k", 16.0);
                let hl = get("hl", 4.0);
                if pos < k {
                    1.0
                } else {
                    0.5f32.powf((pos - k) / hl)
                }
            }
            other => panic!("unknown variant kind '{other}'"),
        };
        v.max(0.0)
    })
}

/// Mirrors generator::generate_with_timer, but with an explicit weight table
/// and a seeded RNG so passes are reproducible per (variant, pass).
fn generate_pass(
    model: &mut Model,
    weights: &[f32; NUMBER_OF_POSITIONS],
    prompt: &str,
    n: usize,
    seed: u64,
) -> (Vec<Token>, usize, f64) {
    model.weights = *weights;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tok = tokenize(prompt);
    let prompt_len = tok.tokens().len();

    let start = Instant::now();
    for _ in 0..n {
        let mut scores: HashMap<&str, f32> = HashMap::new();
        for (i, t) in tok
            .tokens()
            .iter()
            .rev()
            .take(NUMBER_OF_POSITIONS)
            .enumerate()
        {
            for (w, wt) in model.followers(&t.content, i as u16) {
                *scores.entry(w).or_insert(0.0) += wt;
            }
        }

        let word = {
            let total: f64 = scores.values().map(|&v| v as f64).sum();
            if total > 0.0 {
                let mut x = rng.random::<f64>() * total;
                let mut chosen: Option<&str> = None;
                for (w, &wt) in scores.iter() {
                    x -= wt as f64;
                    if x <= 0.0 {
                        chosen = Some(w);
                        break;
                    }
                }
                chosen.unwrap_or_else(|| scores.keys().next().unwrap()).to_string()
            } else if !scores.is_empty() {
                let idx = rng.random_range(0..scores.len());
                scores.keys().nth(idx).unwrap().to_string()
            } else {
                "NO_WORD_FOUND".to_string()
            }
        };
        drop(scores);

        let token = Token {
            content: word.clone(),
            mask: Mask::classify_word(&word),
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
    ppl: f64,
    repeat4: f64,
    ttr: f64,
    pos0: f64,
    avgsent: f64,
    oov: f64,
}

fn compute_metrics(model: &Model, lm: &EvalLM, full: &[Token], prompt_len: usize) -> Metrics {
    let nonws: Vec<&Token> = full.iter().filter(|t| t.mask != Mask::Whitespace).collect();
    // generated region, keeping the last prompt token as context for pair stats
    let gen_start = prompt_len.saturating_sub(1).min(nonws.len());
    let gen_toks = &nonws[gen_start..];

    // ppl over generated non-ws tokens
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

    // repeat4 over generated non-ws 4-grams
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
    let repeat4 = if total4 > 0 {
        dup4 as f64 / total4 as f64
    } else {
        0.0
    };

    // word-level stats
    let words: Vec<&&Token> = gen_toks.iter().filter(|t| t.mask == Mask::None).collect();
    let mut uniq: HashSet<&str> = HashSet::new();
    let mut oov = 0usize;
    for t in &words {
        uniq.insert(t.content.as_str());
        if !lm.vocab.contains_key(t.content.as_str()) {
            oov += 1;
        }
    }
    let ttr = if words.is_empty() {
        0.0
    } else {
        uniq.len() as f64 / words.len() as f64
    };
    let oov = if words.is_empty() {
        0.0
    } else {
        oov as f64 / words.len() as f64
    };

    // sentences: terminal punctuation ends a sentence
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

    // pos0 validity over adjacent generated pairs
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
    let pos0 = if total_pairs > 0 {
        valid as f64 / total_pairs as f64
    } else {
        0.0
    };

    Metrics {
        ppl,
        repeat4,
        ttr,
        pos0,
        avgsent,
        oov,
    }
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    (m, var.sqrt())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!(
            "usage: {} <model> <tokens_per_pass> <passes> <eval_jsonl> <eval_tail_lines> <out_prefix> <variant_spec>...",
            args[0]
        );
        std::process::exit(1);
    }
    let model_name = &args[1];
    let tokens_per_pass: usize = args[2].parse().unwrap();
    let passes: usize = args[3].parse().unwrap();
    let eval_jsonl = &args[4];
    let eval_tail: usize = args[5].parse().unwrap();
    let out_prefix = &args[6];
    let specs = &args[7..];

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    writeln!(
        out,
        "pass\tvariant\tppl\trepeat4\tttr\tpos0\tavgsent\toov\tntok\tsecs\trss_mb"
    )
    .unwrap();

    eprintln!("building eval LM from last {eval_tail} lines of {eval_jsonl}...");
    let start = Instant::now();
    let (lm, eval_lines) = build_eval_lm(eval_jsonl, eval_tail).unwrap();
    eprintln!(
        "eval LM: {} tokens, vocab {}, bigrams {}, built in {:.1}s, rss {:.0} MB",
        lm.n_tokens,
        lm.vocab.len(),
        lm.bi.len(),
        start.elapsed().as_secs_f64(),
        rss_mb()
    );

    // reference: score the first 1000 held-out lines under the eval LM itself
    let mut ref_ids: Vec<u32> = Vec::new();
    for line in eval_lines.iter().take(1000) {
        if let Ok(t) = serde_json::from_str::<TextLine>(line) {
            for tok in tokenize(&t.text).tokens() {
                if tok.mask != Mask::Whitespace {
                    ref_ids.push(lm.lookup(&tok.content));
                }
            }
        }
    }
    let mut nll = 0.0;
    for w in ref_ids.windows(2) {
        nll += -lm.prob(w[0], w[1]).ln();
    }
    let ref_ppl = (nll / (ref_ids.len() - 1) as f64).exp();
    eprintln!(
        "reference: eval-corpus self-ppl over {} tokens = {:.2}, rss {:.0} MB",
        ref_ids.len(),
        ref_ppl,
        rss_mb()
    );

    eprintln!("loading model '{model_name}'...");
    let mut model = Model::load(model_name).unwrap();
    eprintln!("model loaded, rss {:.0} MB", rss_mb());

    struct Row {
        variant: String,
        m: Metrics,
        secs: f64,
    }
    let mut rows: Vec<Row> = Vec::new();

    for spec in specs {
        let weights = parse_variant(spec);
        for pass in 0..passes {
            let prompt = PROMPTS[pass % PROMPTS.len()];
            let seed = fnv(spec, pass as u64);
            let (full, prompt_len, secs) =
                generate_pass(&mut model, &weights, prompt, tokens_per_pass, seed);
            let m = compute_metrics(&model, &lm, &full, prompt_len);
            let rss = rss_mb();
            writeln!(
                out,
                "pass\t{spec}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{}\t{:.1}\t{:.0}",
                m.ppl,
                m.repeat4,
                m.ttr,
                m.pos0,
                m.avgsent,
                m.oov,
                full.len() - prompt_len,
                secs,
                rss
            )
            .unwrap();
            out.flush().unwrap();

            if pass == 0 {
                let safe: String = spec
                    .chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect();
                let path = format!("{out_prefix}_{safe}.txt");
                let mut text = String::new();
                for (i, t) in full.iter().enumerate() {
                    text.push_str(&t.content);
                    let next_is_word = full
                        .get(i + 1)
                        .is_some_and(|n| n.mask == Mask::None);
                    if (t.mask == Mask::None && next_is_word) || t.mask == Mask::Punctuation {
                        text.push(' ');
                    }
                }
                std::fs::write(path, text).unwrap();
            }

            rows.push(Row {
                variant: spec.clone(),
                m,
                secs,
            });

            if rss > 16_000.0 {
                eprintln!("WARNING: rss {:.0} MB over 16 GB, aborting bench", rss);
                writeln!(out, "abort\trss_limit\t{rss:.0}").unwrap();
                out.flush().unwrap();
                std::process::exit(2);
            }
        }
    }

    // per-variant summary
    writeln!(
        out,
        "sum\tvariant\tppl_mean\tppl_std\trepeat4_mean\trepeat4_std\tpos0_mean\tpos0_std\tttr_mean\tavgsent_mean\toov_mean\tsecs_total"
    )
    .unwrap();
    let mut summaries: Vec<(String, [f64; 6], f64)> = Vec::new();
    for spec in specs {
        let rs: Vec<&Row> = rows.iter().filter(|r| r.variant == *spec).collect();
        let col = |f: fn(&Metrics) -> f64| rs.iter().map(|r| f(&r.m)).collect::<Vec<_>>();
        let ppl = col(|m| m.ppl);
        let rep = col(|m| m.repeat4);
        let pos = col(|m| m.pos0);
        let ttr = col(|m| m.ttr);
        let sent = col(|m| m.avgsent);
        let oov = col(|m| m.oov);
        let secs_total: f64 = rs.iter().map(|r| r.secs).sum();
        let (pm, ps) = mean_std(&ppl);
        let (rm, rs_) = mean_std(&rep);
        let (qm, qs) = mean_std(&pos);
        let (tm, _) = mean_std(&ttr);
        let (sm, _) = mean_std(&sent);
        let (om, _) = mean_std(&oov);
        writeln!(
            out,
            "sum\t{spec}\t{pm:.2}\t{ps:.2}\t{rm:.4}\t{rs_:.4}\t{qm:.4}\t{qs:.4}\t{tm:.4}\t{sm:.2}\t{om:.4}\t{secs_total:.1}"
        )
        .unwrap();
        summaries.push((spec.clone(), [pm, rm, qm, tm, sm, om], secs_total));
    }
    out.flush().unwrap();

    // ranking: by ppl asc; composite = mean of per-metric ranks
    // (ppl asc, repeat4 asc, pos0 desc, |avgsent - ref| asc, ttr asc)
    let ref_sent = 11.0; // rough corpus average; only used for the composite
    let mut ranked = summaries.clone();
    ranked.sort_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap());
    let rank_of = |vals: Vec<f64>, desc: bool| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..vals.len()).collect();
        idx.sort_by(|&a, &b| {
            let (x, y) = (vals[a], vals[b]);
            if desc {
                y.partial_cmp(&x).unwrap()
            } else {
                x.partial_cmp(&y).unwrap()
            }
        });
        let mut r = vec![0usize; vals.len()];
        for (pos, i) in idx.into_iter().enumerate() {
            r[i] = pos;
        }
        r
    };
    let ppls: Vec<f64> = ranked.iter().map(|s| s.1[0]).collect();
    let reps: Vec<f64> = ranked.iter().map(|s| s.1[1]).collect();
    let poss: Vec<f64> = ranked.iter().map(|s| s.1[2]).collect();
    let sents: Vec<f64> = ranked.iter().map(|s| (s.1[4] - ref_sent).abs()).collect();
    let ttrs: Vec<f64> = ranked.iter().map(|s| s.1[3]).collect();
    let rp = rank_of(ppls.clone(), false);
    let rr = rank_of(reps, false);
    let rq = rank_of(poss, true);
    let rs2 = rank_of(sents, false);
    let rt = rank_of(ttrs, false);

    writeln!(
        out,
        "rank\tvariant\tppl_mean\trepeat4_mean\tpos0_mean\tttr_mean\tavgsent_mean\toov_mean\tcomposite_rank\tppl_rank"
    )
    .unwrap();
    for (i, (name, m, _)) in ranked.iter().enumerate() {
        let composite = (rp[i] + rr[i] + rq[i] + rs2[i] + rt[i]) as f64 / 5.0;
        writeln!(
            out,
            "rank\t{name}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.2}\t{:.4}\t{composite:.1}\t{}",
            m[0], m[1], m[2], m[3], m[4], m[5], rp[i]
        )
        .unwrap();
    }
    out.flush().unwrap();
    eprintln!("done. peak-logged rss at end: {:.0} MB", rss_mb());
}
