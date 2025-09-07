use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ahash::AHashMap;
use memmap2::Mmap;
use rayon::prelude::*;
use regex::Regex;

#[derive(Clone)]
pub struct PreTrainedBPE {
    pub merges: AHashMap<(u32, u32), u32>,
    pub ranks: AHashMap<(u32, u32), usize>,
    pub id_to_bytes: Vec<Vec<u8>>,
    pub merges_ordered: Vec<((u32, u32), u32)>,
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

fn pretokenize_bytes(bytes: &[u8]) -> AHashMap<Vec<u8>, u32> {
    let mut counts: AHashMap<Vec<u8>, u32> = AHashMap::new();
    let mut i: usize = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            let start = i;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let tok = bytes[start..i].to_vec();
            *counts.entry(tok).or_insert(0) += 1;
        } else {
            let start = i;
            i += 1;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let tok = bytes[start..i].to_vec();
            *counts.entry(tok).or_insert(0) += 1;
        }
    }
    counts
}

fn bytes_map_to_corpus(counts: &AHashMap<Vec<u8>, u32>) -> Vec<(Vec<u32>, u32)> {
    counts
        .iter()
        .map(|(bs, &c)| (bs.iter().map(|&b| b as u32).collect(), c))
        .collect()
}

fn compute_pair_counts(corpus: &[(Vec<u32>, u32)]) -> AHashMap<(u32, u32), u64> {
    corpus
        .par_iter()
        .map(|(syms, freq)| {
            let freq = *freq;
            let mut local: AHashMap<(u32, u32), u64> = AHashMap::new();
            if syms.len() >= 2 {
                for w in syms.windows(2) {
                    let pair = (w[0], w[1]);
                    *local.entry(pair).or_insert(0) += freq as u64;
                }
            }
            local
        })
        .reduce(
            || AHashMap::new(),
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

fn build_ranks(merges_seq: &[((u32, u32), u32)]) -> AHashMap<(u32, u32), usize> {
    let mut r = AHashMap::new();
    for (i, (p, _)) in merges_seq.iter().enumerate() {
        r.insert(*p, i);
    }
    r
}

fn apply_best_merge_once(
    seq: &[u32],
    ranks: &AHashMap<(u32, u32), usize>,
    merges: &AHashMap<(u32, u32), u32>,
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
    let bytes = text.as_bytes();
    let mut i: usize = 0;
    while i < bytes.len() {
        let start = i;
        let is_ws = bytes[i].is_ascii_whitespace();
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() == is_ws {
            i += 1;
        }
        let mut seq: Vec<u32> = bytes[start..i].iter().map(|&b| b as u32).collect();
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
) -> anyhow::Result<PreTrainedBPE> {
    let merges_fp = dir.as_ref().join("merges.txt");
    let f = fs::File::open(&merges_fp)?;
    let reader = BufReader::new(f);

    let mut merges_ordered: Vec<((u32, u32), u32)> = Vec::new();
    let mut merges: AHashMap<(u32, u32), u32> = AHashMap::new();
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
    })
}
