use std::env;

use jump_consistent_hash::jump_consistent_hash;

fn main() {
  let args: Vec<String> = env::args().collect();

  if args.len() < 3 {
    println!("usage: {} <key> <buckets>", args[0]);
    let key = 42;
    let buckets = 10;
    println!("example: hash({}, {}) = {}", key, buckets, jump_consistent_hash(key, buckets));
    return;
  }

  let key = args[1].parse::<u64>().expect("key must be u64");
  let buckets = args[2].parse::<i32>().expect("buckets must be i32");
  let result = jump_consistent_hash(key, buckets);
  println!("{}", result);
}
