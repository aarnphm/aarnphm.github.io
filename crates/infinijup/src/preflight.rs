use anyhow::{bail, Context, Result};
use nix::unistd::Pid;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct PreflightConfig {
  pub require_no_uvm: bool,
  pub require_no_ipc: bool,
}

impl Default for PreflightConfig {
  fn default() -> Self {
    Self { require_no_uvm: true, require_no_ipc: true }
  }
}

#[derive(Debug, Default, Clone)]
pub struct PreflightReport {
  pub uvm_loaded: bool,
  pub uvm_device_open: bool,
  pub ipc_risk: bool,
  pub gpu_device_users: usize,
  pub messages: Vec<String>,
}

pub fn enforce_no_uvm_ipc(root_pid: i32, cfg: PreflightConfig) -> Result<PreflightReport> {
  let pid = Pid::from_raw(root_pid);
  let mut r = check_uvm_ipc(pid)?;

  if cfg.require_no_uvm {
    if r.uvm_device_open {
      r.messages.push("/dev/nvidia-uvm is open by target subtree".into());
      bail!("UVM usage detected: refusing to checkpoint (P0 policy)");
    }
  }
  if cfg.require_no_ipc {
    if r.ipc_risk || r.gpu_device_users > 1 {
      r.messages.push("Multiple processes use GPU devices; IPC risk".into());
      bail!("CUDA IPC risk detected: refusing to checkpoint (P0 policy)");
    }
  }
  Ok(r)
}

fn check_uvm_ipc(root: Pid) -> Result<PreflightReport> {
  let mut rep = PreflightReport::default();

  //XXX: This is essentially some heuristic we rely on. Brittle builtin, hopefully CUDA doesn't
  //fuck us over.

  // is nvidia_uvm module loaded?
  let modules = fs::read_to_string("/proc/modules").unwrap_or_default();
  if modules.contains("nvidia_uvm") {
    rep.uvm_loaded = true;
  }

  // Gather subtree PIDs (best-effort): include root and children from /proc/<pid>/task/*/children
  let mut pids = vec![root.as_raw()];
  if let Ok(children) = read_children_recursive(root.as_raw()) {
    pids.extend(children);
  }

  // Heuristic 2: treat any process with an open /dev/nvidia-uvm fd as UVM-in-use
  let mut gpu_users = 0usize;
  for p in pids.iter().copied() {
    if has_open_nvidia_fd(p) {
      gpu_users += 1;
    }
    if has_open_uvm_fd(p) {
      rep.uvm_device_open = true;
    }
    if maps_has_sysv_or_shm(p) {
      // Coarse signal for potential IPC; many apps use SYSV/SHM, so only mark risk.
      rep.ipc_risk = true;
    }
  }
  rep.gpu_device_users = gpu_users;
  Ok(rep)
}

fn read_children_recursive(pid: i32) -> Result<Vec<i32>> {
  let mut acc = Vec::new();
  let task_dir = PathBuf::from(format!("/proc/{pid}/task"));
  let tasks = fs::read_dir(&task_dir).with_context(|| format!("read_dir {task_dir:?}"))?;
  let mut all_children = Vec::new();
  for t in tasks.flatten() {
    let path = t.path().join("children");
    if let Ok(s) = fs::read_to_string(&path) {
      for tok in s.split_whitespace() {
        if let Ok(cpid) = tok.parse::<i32>() { all_children.push(cpid); }
      }
    }
  }
  for c in all_children.clone() { // DFS
    acc.push(c);
    if let Ok(mut sub) = read_children_recursive(c) { acc.append(&mut sub); }
  }
  acc.sort_unstable();
  acc.dedup();
  Ok(acc)
}

fn has_open_uvm_fd(pid: i32) -> bool { has_open_dev(pid, "nvidia-uvm") }
fn has_open_nvidia_fd(pid: i32) -> bool { has_open_dev(pid, "nvidia") }

fn has_open_dev(pid: i32, name: &str) -> bool {
  let fd_dir = PathBuf::from(format!("/proc/{pid}/fd"));
  if let Ok(iter) = fs::read_dir(&fd_dir) {
    for e in iter.flatten() {
      if let Ok(target) = fs::read_link(e.path()) {
        if let Some(s) = target.to_str() {
          if s.contains("/dev/") && s.contains(name) { return true; }
        }
      }
    }
  }
  false
}

fn maps_has_sysv_or_shm(pid: i32) -> bool {
  let maps = PathBuf::from(format!("/proc/{pid}/maps"));
  if let Ok(s) = fs::read_to_string(&maps) {
    // Coarse indicator of shared mem usage; not CUDA-specific but useful for a conservative block.
    if s.contains("/dev/shm") || s.contains("SYSV") { return true; }
  }
  false
}

