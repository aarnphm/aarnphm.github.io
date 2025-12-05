use std::fs;

fn parse_input(input: &str) -> Vec<(u64, u64)> {
    input
        .split("\n\n")
        .next()
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let parts: Vec<u64> = l.split('-').map(|s| s.parse().unwrap()).collect();
            (parts[0], parts[1])
        })
        .collect()
}

fn parse_ids(input: &str) -> Vec<u64> {
    input
        .split("\n\n")
        .nth(1)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.parse().unwrap())
        .collect()
}

fn merge(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    if ranges.is_empty() {
        return ranges;
    }

    ranges.sort_by_key(|r| r.0);
    let mut merged = vec![ranges[0]];

    for (start, end) in ranges.into_iter().skip(1) {
        let last = merged.last_mut().unwrap();
        if start <= last.1 + 1 {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }

    merged
}

fn p1(ranges: &[(u64, u64)], ids: &[u64]) -> usize {
    ids.iter()
        .filter(|&&id| ranges.iter().any(|&(s, e)| id >= s && id <= e))
        .count()
}

fn p2(ranges: Vec<(u64, u64)>) -> u64 {
    merge(ranges)
        .iter()
        .map(|(start, end)| end - start + 1)
        .sum()
}

fn main() {
    let input = fs::read_to_string("d5.txt").expect("Failed to read file");

    let ranges = parse_input(&input);
    let ids = parse_ids(&input);

    println!("p1: {}", p1(&ranges, &ids));
    println!("p2: {}", p2(ranges));
}
