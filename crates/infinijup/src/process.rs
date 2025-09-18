use anyhow::Result;
use std::fs;
use std::path::PathBuf;

pub fn subtree_pids(root_pid: i32) -> Result<Vec<i32>> {
  let mut acc = vec![root_pid];
  let tasks_dir = PathBuf::from(format!("/proc/{root_pid}/task"));
  if let Ok(tasks) = fs::read_dir(&tasks_dir) {
    for t in tasks.flatten() {
      let children_path = t.path().join("children");
      if let Ok(s) = fs::read_to_string(&children_path) {
        for tok in s.split_whitespace() { if let Ok(pid) = tok.parse::<i32>() { acc.push(pid); } }
      }
    }
  }
  acc.sort_unstable();
  acc.dedup();
  Ok(acc)
}

