use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context};
use fancy_regex::Regex;
use rustc_hash::FxHashMap;

type Merge = ((u32, u32), u32);

#[derive(Clone)]
pub struct PreTrainedBPE {
    pub merges: FxHashMap<(u32, u32), u32>,
    pub ranks: FxHashMap<(u32, u32), usize>,
    pub id_to_bytes: FxHashMap<u32, Vec<u8>>,
    pub merges_ordered: Vec<Merge>,
    pub pattern: Regex,
}

fn default_pattern() -> &'static str {
    // Keep minibpe.patterns.PRETOKENIZER_PATTERN in sync.
    "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
}

pub fn encode_str(model: &PreTrainedBPE, text: &str) -> anyhow::Result<Vec<u32>> {
    let mut tokens = Vec::new();
    for matched in model.pattern.find_iter(text) {
        tokens.extend(encode_bytes(model, matched?.as_str().as_bytes()));
    }
    Ok(tokens)
}

pub fn encode_bytes(model: &PreTrainedBPE, data: &[u8]) -> Vec<u32> {
    let mut seq = data.iter().map(|&byte| u32::from(byte)).collect();
    byte_pair_merge(&mut seq, &model.ranks, &model.merges);
    seq
}

pub fn decode_bytes(model: &PreTrainedBPE, ids: &[u32]) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for id in ids {
        let part = model
            .id_to_bytes
            .get(id)
            .with_context(|| format!("unknown token id: {id}"))?;
        bytes.extend_from_slice(part);
    }
    Ok(bytes)
}

pub fn decode_ids(model: &PreTrainedBPE, ids: &[u32]) -> anyhow::Result<String> {
    Ok(String::from_utf8_lossy(&decode_bytes(model, ids)?).into_owned())
}

fn from_merges(mut merges_ordered: Vec<Merge>) -> anyhow::Result<PreTrainedBPE> {
    merges_ordered.sort_by_key(|(_, id)| *id);
    let mut merges = FxHashMap::default();
    let mut ranks = FxHashMap::default();
    let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = (0u8..=255)
        .map(|byte| (u32::from(byte), vec![byte]))
        .collect();
    for (rank, ((a, b), id)) in merges_ordered.iter().enumerate() {
        if id_to_bytes.contains_key(id) {
            bail!("duplicate or reserved token id: {id}");
        }
        if merges.contains_key(&(*a, *b)) {
            bail!("duplicate merge pair: ({a}, {b})");
        }
        let left = id_to_bytes
            .get(a)
            .with_context(|| format!("merge refers to an undefined token: {a}"))?;
        let right = id_to_bytes
            .get(b)
            .with_context(|| format!("merge refers to an undefined token: {b}"))?;
        let mut bytes = Vec::with_capacity(left.len() + right.len());
        bytes.extend_from_slice(left);
        bytes.extend_from_slice(right);
        id_to_bytes.insert(*id, bytes);
        merges.insert((*a, *b), *id);
        ranks.insert((*a, *b), rank);
    }
    Ok(PreTrainedBPE {
        merges,
        ranks,
        id_to_bytes,
        merges_ordered,
        pattern: Regex::new(default_pattern())?,
    })
}

fn read_merges(reader: impl BufRead) -> anyhow::Result<PreTrainedBPE> {
    let mut merges = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let normalized = line.replace(',', " ");
        let parts: Vec<&str> = normalized.split_whitespace().collect();
        let [a, b, id] = parts.as_slice() else {
            bail!("expected three merge ids: {line}");
        };
        merges.push(((a.parse()?, b.parse()?), id.parse()?));
    }
    from_merges(merges)
}

pub fn load_from_dir(dir: impl AsRef<Path>) -> anyhow::Result<PreTrainedBPE> {
    read_merges(BufReader::new(fs::File::open(
        dir.as_ref().join("merges.txt"),
    )?))
}

/// Apply the lowest-ranked available pair, resolving overlaps left to right.
/// Each step scans the remaining sequence, so a pretoken takes O(n^2) time
/// in the worst case. No throughput claim follows from this implementation.
fn byte_pair_merge(
    seq: &mut Vec<u32>,
    ranks: &FxHashMap<(u32, u32), usize>,
    merges: &FxHashMap<(u32, u32), u32>,
) {
    loop {
        let best = seq
            .windows(2)
            .enumerate()
            .filter_map(|(index, pair)| ranks.get(&(pair[0], pair[1])).map(|rank| (*rank, index)))
            .min();
        let Some((_, index)) = best else { break };
        let Some(&id) = merges.get(&(seq[index], seq[index + 1])) else {
            break;
        };
        seq[index] = id;
        seq.remove(index + 1);
    }
}

#[cfg(test)]
mod tests {
    use crate::bpe::{
        decode_bytes, decode_ids, encode_bytes, encode_str, from_merges, read_merges,
    };
    use std::io::Cursor;

    #[test]
    fn arbitrary_bytes_and_unicode_roundtrip() {
        let model = from_merges(vec![((0, 255), 256), ((195, 169), 257)]).unwrap();
        let bytes: Vec<u8> = (0u8..=255).chain([0, 255, 0, 255]).collect();
        assert_eq!(
            decode_bytes(&model, &encode_bytes(&model, &bytes)).unwrap(),
            bytes
        );
        for text in ["", "café", "你好 🦀", "a  b\n\t c", "e\u{301}", "I'm we'll"] {
            assert_eq!(
                decode_ids(&model, &encode_str(&model, text).unwrap()).unwrap(),
                text
            );
        }
    }

    #[test]
    fn merge_rank_wins_over_leftmost_pair() {
        let model = from_merges(vec![((98, 99), 256), ((97, 98), 257)]).unwrap();
        assert_eq!(encode_bytes(&model, b"abc"), vec![97, 256]);
    }

    #[test]
    fn nested_and_overlapping_merges() {
        let model = from_merges(vec![((97, 97), 256), ((256, 97), 257)]).unwrap();
        assert_eq!(encode_bytes(&model, b"aaaaa"), vec![256, 257]);
        assert_eq!(decode_bytes(&model, &[256, 257]).unwrap(), b"aaaaa");
    }

    #[test]
    fn whitespace_pretokenization_matches_python() {
        let model = from_merges(vec![((32, 32), 256)]).unwrap();
        assert_eq!(encode_str(&model, "a  b").unwrap(), vec![97, 32, 32, 98]);
        assert_eq!(encode_bytes(&model, b"a  b"), vec![97, 256, 98]);
    }

    #[test]
    fn load_reconstructs_utf8_fragments_without_vocab() {
        let model = read_merges(Cursor::new("256,172,257\n226,130,256\n")).unwrap();
        assert_eq!(decode_bytes(&model, &[256]).unwrap(), [226, 130]);
        assert_eq!(decode_ids(&model, &[257]).unwrap(), "€");
    }

    #[test]
    fn malformed_models_and_unknown_ids_are_errors() {
        for text in [
            "97,98\n",
            "no,98,256\n",
            "300,98,256\n",
            "97,98,97\n",
            "97,98,4294967296\n",
            "97,98,256\n98,99,256\n",
            "97,98,256\n97,98,257\n",
        ] {
            assert!(read_merges(Cursor::new(text)).is_err(), "{text}");
        }
        let model = from_merges(vec![]).unwrap();
        assert!(decode_bytes(&model, &[256]).is_err());
    }
}
