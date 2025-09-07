use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use rustc_hash::FxHashMap;
use fancy_regex::Regex;

#[derive(Clone)]
pub struct PreTrainedBPE {
    pub merges: FxHashMap<(u32, u32), u32>,
    pub ranks: FxHashMap<(u32, u32), usize>,
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

fn build_ranks(merges_seq: &[((u32, u32), u32)]) -> FxHashMap<(u32, u32), usize> {
    let mut r = FxHashMap::default();
    for (i, (p, _)) in merges_seq.iter().enumerate() {
        r.insert(*p, i);
    }
    r
}

pub fn encode_str(model: &PreTrainedBPE, text: &str) -> Vec<u32> {
    let mut tokens: Vec<u32> = Vec::new();
    for t in model
        .pattern
        .find_iter(text)
        .map(|m| m.expect("regex match").as_str())
    {
        let mut seq: Vec<u32> = t.as_bytes().iter().map(|&b| b as u32).collect();
        byte_pair_merge(&mut seq, &model.ranks, &model.merges);
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
    let regex = compile_with_fallback(None);
    let merges_fp = dir.as_ref().join("merges.txt");
    let f = fs::File::open(&merges_fp)?;
    let reader = BufReader::new(f);

    let mut merges_ordered: Vec<((u32, u32), u32)> = Vec::new();
    let mut merges: FxHashMap<(u32, u32), u32> = FxHashMap::default();
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

// In-place tiktoken-like byte pair merge using ranks ordering
fn byte_pair_merge(seq: &mut Vec<u32>, ranks: &FxHashMap<(u32, u32), usize>, merges: &FxHashMap<(u32, u32), u32>) {
    if seq.len() < 2 { return; }
    loop {
        let mut best: Option<(usize, usize)> = None; // (rank, index)
        let mut best_rank: usize = usize::MAX;
        let mut idx: usize = 0;
        while idx + 1 < seq.len() {
            if let Some(&r) = ranks.get(&(seq[idx], seq[idx + 1])) {
                if r < best_rank { best_rank = r; best = Some((r, idx)); }
            }
            idx += 1;
        }
        let Some((_, i)) = best else { break };
        let new_id = match merges.get(&(seq[i], seq[i + 1])) { Some(x) => *x, None => break };
        // merge at i
        seq[i] = new_id;
        seq.remove(i + 1);
        if seq.len() < 2 { break; }
    }
}
