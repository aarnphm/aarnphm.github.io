use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use hashbrown::HashMap as HbMap;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use regex::Regex;

#[derive(Clone)]
pub struct PreTrainedBPE {
    pub merges: HbMap<(u32, u32), u32>,
    pub ranks: HbMap<(u32, u32), usize>,
    pub id_to_bytes: Vec<Vec<u8>>,
    pub merges_ordered: Vec<((u32, u32), u32)>,
    pub pattern: Regex,
}

pub(crate) fn compile_with_fallback(user: Option<&str>) -> Regex {
    let fallback = default_pattern();
    let pat = user.unwrap_or(fallback);
    match Regex::new(pat) {
        Ok(r) => r,
        Err(_) => Regex::new(fallback).expect("fallback regex must compile"),
    }
}

fn default_pattern() -> &'static str {
    "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+"
}

fn pretokenize_str(pattern: &Regex, text: &str) -> HbMap<String, u32> {
    let mut local: HbMap<String, u32> = HbMap::new();
    for m in pattern.find_iter(text) {
        *local.entry(m.as_str().to_string()).or_insert(0) += 1;
    }
    local
}

fn count_tokens_parallel(pattern: &Regex, paths: Vec<String>) -> HbMap<String, u32> {
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("{msg} {bar:40.cyan/blue} {pos}/{len} ({eta})").unwrap(),
    );
    pb.set_message("Pretokenizing");
    let result = paths
        .into_par_iter()
        .map(|p| {
            let file = match fs::File::open(&p) {
                Ok(f) => f,
                Err(_) => return HbMap::new(),
            };
            let mmap = unsafe { Mmap::map(&file).unwrap() };
            let bytes = &mmap[..];
            let chunk_size: usize = 2 * 1024 * 1024; // 2MB chunks
            let mut totals: HbMap<String, u32> = HbMap::new();
            // find UTF-8 boundaries to avoid splitting in the middle of a codepoint
            let mut start: usize = 0;
            while start < bytes.len() {
                let end_guess = start.saturating_add(chunk_size).min(bytes.len());
                let mut end = end_guess;
                while end < bytes.len() && (bytes[end] & 0b1100_0000) == 0b1000_0000 {
                    end += 1;
                }
                let s = std::str::from_utf8(&bytes[start..end]).unwrap_or("");
                let local = pretokenize_str(pattern, s);
                for (k, v) in local {
                    *totals.entry(k).or_insert(0) += v;
                }
                start = end;
            }
            pb.inc(1);
            totals
        })
        .reduce(
            || HbMap::new(),
            |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            },
        );
    pb.finish_and_clear();
    result
}

fn strings_to_corpus(counts: &HbMap<String, u32>) -> Vec<(Vec<u32>, u32)> {
    counts
        .iter()
        .map(|(s, &c)| (s.as_bytes().iter().map(|&b| b as u32).collect(), c))
        .collect()
}

fn compute_pair_counts(corpus: &[(Vec<u32>, u32)]) -> HbMap<(u32, u32), u64> {
    corpus
        .par_iter()
        .map(|(syms, freq)| {
            let freq = *freq;
            let mut local: HbMap<(u32, u32), u64> = HbMap::new();
            if syms.len() >= 2 {
                for w in syms.windows(2) {
                    let pair = (w[0], w[1]);
                    *local.entry(pair).or_insert(0) += freq as u64;
                }
            }
            local
        })
        .reduce(
            || HbMap::new(),
            |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            },
        )
}

fn merge_corpus(corpus: &mut [(Vec<u32>, u32)], pair: (u32, u32), new_id: u32) {
    corpus.par_iter_mut().for_each(|(syms, _)| {
        if syms.len() < 2 {
            return;
        }
        let mut merged: Vec<u32> = Vec::with_capacity(syms.len());
        let mut i = 0;
        while i < syms.len() {
            if i + 1 < syms.len() && syms[i] == pair.0 && syms[i + 1] == pair.1 {
                merged.push(new_id);
                i += 2;
            } else {
                merged.push(syms[i]);
                i += 1;
            }
        }
        *syms = merged;
    });
}

pub fn train_from_files(
    paths: Vec<String>,
    num_merges: usize,
    pattern: Option<String>,
    num_threads: Option<usize>,
) -> PreTrainedBPE {
    if let Some(n) = num_threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
    let regex = compile_with_fallback(pattern.as_deref());
    let counts = count_tokens_parallel(&regex, paths);
    let (merges_seq, id_to_bytes) = train_bpe_internal(counts, num_merges);
    let mut merges: HbMap<(u32, u32), u32> = HbMap::new();
    for (pair, nid) in merges_seq.iter() {
        merges.insert(*pair, *nid);
    }
    let ranks = build_ranks(&merges_seq);
    PreTrainedBPE {
        merges,
        ranks,
        id_to_bytes,
        merges_ordered: merges_seq,
        pattern: regex,
    }
}

fn train_bpe_internal(
    token_counts: HbMap<String, u32>,
    num_merges: usize,
) -> (Vec<((u32, u32), u32)>, Vec<Vec<u8>>) {
    let mut id_to_bytes: Vec<Vec<u8>> = (0u32..256).map(|i| vec![i as u8]).collect();
    let mut corpus = strings_to_corpus(&token_counts);
    let mut merges_seq: Vec<((u32, u32), u32)> = Vec::new();
    let mut next_id: u32 = 256;
    let pb = ProgressBar::new(num_merges as u64);
    pb.set_style(
        ProgressStyle::with_template("{msg} {bar:40.magenta/blue} {pos}/{len} ({eta})").unwrap(),
    );
    pb.set_message("BPE merges");
    for _ in 0..num_merges {
        let pair_counts = compute_pair_counts(&corpus);
        if pair_counts.is_empty() {
            break;
        }
        let best = pair_counts.into_iter().max_by_key(|(_, c)| *c).unwrap().0;
        let new_id = next_id;
        next_id += 1;
        merge_corpus(&mut corpus, best, new_id);
        let mut bytes = Vec::with_capacity(
            id_to_bytes[best.0 as usize].len() + id_to_bytes[best.1 as usize].len(),
        );
        bytes.extend_from_slice(&id_to_bytes[best.0 as usize]);
        bytes.extend_from_slice(&id_to_bytes[best.1 as usize]);
        id_to_bytes.push(bytes);
        merges_seq.push((best, new_id));
        pb.inc(1);
    }
    pb.finish_and_clear();
    (merges_seq, id_to_bytes)
}

pub(crate) fn train_bpe_internal_private(
    token_counts: HbMap<String, u32>,
    num_merges: usize,
) -> (Vec<((u32, u32), u32)>, Vec<Vec<u8>>) {
    train_bpe_internal(token_counts, num_merges)
}

fn build_ranks(merges_seq: &[((u32, u32), u32)]) -> HbMap<(u32, u32), usize> {
    let mut r = HbMap::new();
    for (i, (p, _)) in merges_seq.iter().enumerate() {
        r.insert(*p, i);
    }
    r
}

pub(crate) fn build_ranks_private(merges_seq: &[((u32, u32), u32)]) -> HbMap<(u32, u32), usize> {
    build_ranks(merges_seq)
}

fn apply_best_merge_once(
    seq: &[u32],
    ranks: &HbMap<(u32, u32), usize>,
    merges: &HbMap<(u32, u32), u32>,
) -> Option<Vec<u32>> {
    if seq.len() < 2 {
        return None;
    }
    let mut best: Option<((u32, u32), usize)> = None;
    for w in seq.windows(2) {
        let pair = (w[0], w[1]);
        if let Some(&rank) = ranks.get(&pair) {
            match best {
                None => best = Some((pair, rank)),
                Some((_, br)) if rank < br => best = Some((pair, rank)),
                _ => {}
            }
        }
    }
    let ((a, b), _) = best?;
    let new_id = *merges.get(&(a, b)).unwrap();
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == a && seq[i + 1] == b {
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    Some(out)
}

pub fn encode_str(model: &PreTrainedBPE, text: &str) -> Vec<u32> {
    let mut tokens: Vec<u32> = Vec::new();
    for t in model.pattern.find_iter(text).map(|m| m.as_str()) {
        let mut seq: Vec<u32> = t.as_bytes().iter().map(|&b| b as u32).collect();
        loop {
            if let Some(next_seq) = apply_best_merge_once(&seq, &model.ranks, &model.merges) {
                if next_seq.len() == seq.len() {
                    break;
                }
                seq = next_seq;
            } else {
                break;
            }
        }
        tokens.extend(seq);
    }
    tokens
}

pub fn decode_ids(model: &PreTrainedBPE, ids: &[u32]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    for id in ids {
        if let Some(b) = model.id_to_bytes.get(*id as usize) {
            bytes.extend_from_slice(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn load_from_dir<P: AsRef<Path>>(
    dir: P,
    pattern: Option<String>,
) -> anyhow::Result<PreTrainedBPE> {
    let regex = compile_with_fallback(pattern.as_deref());
    let merges_fp = dir.as_ref().join("merges.txt");
    let f = fs::File::open(&merges_fp)?;
    let reader = BufReader::new(f);
    let mut merges_ordered: Vec<((u32, u32), u32)> = Vec::new();
    let mut merges: HbMap<(u32, u32), u32> = HbMap::new();
    for line in reader.lines() {
        let s = line?;
        let t = s.trim();
        if t.is_empty() {
            continue;
        }
        let parts: Vec<&str> = if t.contains(',') {
            t.split(',').collect()
        } else {
            t.split_whitespace().collect()
        };
        if parts.len() < 3 {
            continue;
        }
        let a: u32 = parts[0].trim().parse().unwrap_or(0);
        let b: u32 = parts[1].trim().parse().unwrap_or(0);
        let nid: u32 = parts[2].trim().parse().unwrap_or(0);
        merges.insert((a, b), nid);
        merges_ordered.push(((a, b), nid));
    }
    merges_ordered.sort_by_key(|(_, nid)| *nid);
    let mut id_to_bytes: Vec<Vec<u8>> = (0u32..256).map(|i| vec![i as u8]).collect();
    for ((a, b), nid) in merges_ordered.iter() {
        let mut bytes =
            Vec::with_capacity(id_to_bytes[*a as usize].len() + id_to_bytes[*b as usize].len());
        bytes.extend_from_slice(&id_to_bytes[*a as usize]);
        bytes.extend_from_slice(&id_to_bytes[*b as usize]);
        if id_to_bytes.len() <= *nid as usize {
            id_to_bytes.resize(*nid as usize + 1, Vec::new());
        }
        id_to_bytes[*nid as usize] = bytes;
    }
    let ranks = build_ranks(&merges_ordered);
    Ok(PreTrainedBPE {
        merges,
        ranks,
        id_to_bytes,
        merges_ordered,
        pattern: regex,
    })
}
