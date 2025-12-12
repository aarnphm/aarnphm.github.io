use crate::codegen;
use crate::parser::{MethodSignature, TestData};
use anyhow::{Context, Result};
use regex::Regex;
use std::path::Path;
use std::process::Command;
use tempfile::tempdir;

#[derive(Debug, Clone)]
pub struct RunResult {
    pub output: String,
    pub time_us: Option<u64>,
    pub memory_kb: Option<u64>,
}

fn parse_time_us(stderr: &str) -> Option<u64> {
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("TIME_US:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

fn parse_max_rss_macos(time_stderr: &str) -> Option<u64> {
    let re = Regex::new(r"(\d+)\s+maximum resident set size").ok()?;
    re.captures(time_stderr)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u64>().ok())
        .map(|bytes| bytes / 1024)
}

fn parse_max_rss_linux(time_stderr: &str) -> Option<u64> {
    let re = Regex::new(r"Maximum resident set size[^:]*:\s*(\d+)").ok()?;
    re.captures(time_stderr)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u64>().ok())
}

pub fn run_cpp(
    _root: &Path,
    solution: &Path,
    sig: &MethodSignature,
    test: &TestData,
    verbose: bool,
) -> Result<RunResult> {
    let harness = codegen::generate_cpp_harness(solution, sig, test)?;

    if verbose {
        println!("{}", "generated harness:");
        for (i, line) in harness.lines().enumerate() {
            println!("{:3} | {}", i + 1, line);
        }
        println!();
    }

    let dir = tempdir()?;
    let harness_path = dir.path().join("harness.cpp");
    let binary_path = dir.path().join("harness");

    std::fs::write(&harness_path, &harness)?;

    let compile = Command::new("g++")
        .args([
            "-std=c++17",
            "-O2",
            "-o",
            binary_path.to_str().unwrap(),
            harness_path.to_str().unwrap(),
        ])
        .output()
        .context("failed to run g++")?;

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        anyhow::bail!("compilation failed:\n{}", stderr);
    }

    let run = if cfg!(target_os = "macos") {
        Command::new("/usr/bin/time")
            .args(["-l", binary_path.to_str().unwrap()])
            .output()
            .context("failed to run with /usr/bin/time")?
    } else {
        Command::new("/usr/bin/time")
            .args(["-v", binary_path.to_str().unwrap()])
            .output()
            .context("failed to run with /usr/bin/time")?
    };

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        anyhow::bail!("runtime error:\n{}", stderr);
    }

    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();

    let time_us = parse_time_us(&stderr);
    let memory_kb = if cfg!(target_os = "macos") {
        parse_max_rss_macos(&stderr)
    } else {
        parse_max_rss_linux(&stderr)
    };

    Ok(RunResult {
        output: stdout,
        time_us,
        memory_kb,
    })
}

pub fn run_rs(
    _root: &Path,
    solution: &Path,
    sig: &MethodSignature,
    test: &TestData,
    verbose: bool,
) -> Result<RunResult> {
    let harness = codegen::generate_rs_harness(solution, sig, test)?;

    if verbose {
        println!("{}", "generated harness:");
        for (i, line) in harness.lines().enumerate() {
            println!("{:3} | {}", i + 1, line);
        }
        println!();
    }

    let dir = tempdir()?;
    let harness_path = dir.path().join("harness.rs");
    let binary_path = dir.path().join("harness");

    std::fs::write(&harness_path, &harness)?;

    let compile = Command::new("rustc")
        .args([
            "-O",
            "-o",
            binary_path.to_str().unwrap(),
            harness_path.to_str().unwrap(),
        ])
        .output()
        .context("failed to run rustc")?;

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        anyhow::bail!("compilation failed:\n{}", stderr);
    }

    let run = if cfg!(target_os = "macos") {
        Command::new("/usr/bin/time")
            .args(["-l", binary_path.to_str().unwrap()])
            .output()
            .context("failed to run with /usr/bin/time")?
    } else {
        Command::new("/usr/bin/time")
            .args(["-v", binary_path.to_str().unwrap()])
            .output()
            .context("failed to run with /usr/bin/time")?
    };

    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        anyhow::bail!("runtime error:\n{}", stderr);
    }

    let stdout = String::from_utf8_lossy(&run.stdout).to_string();
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();

    let time_us = parse_time_us(&stderr);
    let memory_kb = if cfg!(target_os = "macos") {
        parse_max_rss_macos(&stderr)
    } else {
        parse_max_rss_linux(&stderr)
    };

    Ok(RunResult {
        output: stdout,
        time_us,
        memory_kb,
    })
}
