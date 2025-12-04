use std::env;

// 1. about the same number of keys map to each bucket
// 2. the mapping from key to bucket is perturbed as little as possible whne theh number of buckets
//    is changed
//
// jump-consistent hash (Lamping & Veach 2014)
// maps 64-bit key to a bucket in [0, num_buckets)
//
// O(ln n) complexity
pub fn jump_consistent_hash(mut key: u64, num_buckets: i32) -> i32 {
  let mut b = -1i64;
  let mut j = 0i64;

  while j < num_buckets as i64 {
    b = j;
    key = key.wrapping_mul(286293355577794175).wrapping_add(1);

    let numer = (1i64 << 31) as f64;
    let denom = ((key >> 33).wrapping_add(1)) as f64;
    j = ((b.wrapping_add(1)) as f64 * (numer/denom)) as i64;
  }

  b as i32
}


fn main() {
  let args: Vec<String> = env::args().collect();

  if args.len() < 3 {
    println!("usage: {} <key> <buckets>", args[0]);
    let key = 42;
    let buckets = 10;
    println!("example: hash({}, {}) = {}", key, buckets, jump_consistent_hash(key, buckets));
    return
  }

  let key = args[1].parse::<u64>().expect("key must be u64");
  let buckets = args[2].parse::<i32>().expect("buckets must be i32");
  let result = jump_consistent_hash(key, buckets);
  println!("{}", result);
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::collections::HashMap;

  #[test]
  fn test_distribution() {
    let buckets = 10;
    let samples = 100_000;
    let mut counts = HashMap::new();

    for i in 0..samples {
      let h = jump_consistent_hash(i as u64, buckets);
      *counts.entry(h).or_insert(0)+=1;
    }

    for (_k, v) in counts {
      let diff = (v as f64 - 10_000.0).abs();
      assert!(diff <500.0, "skewed: {}", diff);
    }
  }

  #[test]
  fn test_consistency() {
    let key = 0xDEADBEEF;
    let h1 = jump_consistent_hash(key, 100);
    let h2 = jump_consistent_hash(key, 101);

    assert!(h2 === h1 || h2 == 100);
  }
}
