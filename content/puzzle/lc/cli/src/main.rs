mod codegen;
mod parser;
mod runner;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
#[command(name = "lc", about = "leetcode test runner")]
struct Cli {
    #[arg(short, long, global = true)]
    directory: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run {
        file: String,
        #[arg(short, long)]
        test: Option<usize>,
        #[arg(short, long)]
        verbose: bool,
        #[arg(short, long)]
        analyze: bool,
        #[arg(long, default_value = "20,40,80,160")]
        sizes: String,
        #[arg(long, default_value = "3")]
        runs: usize,
    },
    List,
}

fn find_git_root() -> Option<PathBuf> {
    Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()))
}

fn find_lc_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = explicit {
        if dir.join("data").is_dir() {
            return Ok(dir);
        }
        anyhow::bail!("invalid lc directory: {} (no data/ subdir)", dir.display());
    }

    if let Some(git_root) = find_git_root() {
        let default_lc = git_root.join("content/puzzle/lc");
        if default_lc.join("data").is_dir() {
            return Ok(default_lc);
        }
    }

    let mut dir = std::env::current_dir()?;
    loop {
        if dir.join("data").is_dir() && (dir.join("cli").is_dir() || dir.ends_with("lc")) {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }

    anyhow::bail!("could not find lc root. use --directory or run from content/puzzle/lc")
}

fn resolve_solution(root: &Path, file: &str) -> Result<(PathBuf, String)> {
    let file = file.trim();
    let (num, ext) = if file.contains('.') {
        let parts: Vec<&str> = file.rsplitn(2, '.').collect();
        (parts[1], parts[0])
    } else {
        let cpp = root.join(format!("{}.cpp", file));
        let rs = root.join(format!("{}.rs", file));
        if cpp.exists() {
            (file, "cpp")
        } else if rs.exists() {
            (file, "rs")
        } else {
            anyhow::bail!("no solution found for {}", file);
        }
    };
    let path = root.join(format!("{}.{}", num, ext));
    if !path.exists() {
        anyhow::bail!("solution file not found: {}", path.display());
    }
    Ok((path, ext.to_string()))
}

fn find_test_cases(root: &Path, problem: &str) -> Result<Vec<PathBuf>> {
    let data_dir = root.join("data");
    let prefix = format!("{}_", problem);
    let mut cases: Vec<PathBuf> = std::fs::read_dir(&data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&prefix) && n.ends_with(".txt"))
                .unwrap_or(false)
        })
        .collect();
    cases.sort();
    Ok(cases)
}

fn format_time(us: u64) -> String {
    if us >= 1_000_000 {
        format!("{:.2} s", us as f64 / 1_000_000.0)
    } else if us >= 1000 {
        format!("{:.2} ms", us as f64 / 1000.0)
    } else {
        format!("{} μs", us)
    }
}

fn format_memory(kb: u64) -> String {
    if kb >= 1024 {
        format!("{:.2} MB", kb as f64 / 1024.0)
    } else {
        format!("{} KB", kb)
    }
}

struct MethodStats {
    name: String,
    passed: usize,
    failed: usize,
    total_time_us: u64,
    max_memory_kb: u64,
    time_samples: usize,
}

fn run_tests(
    root: &Path,
    solution: &Path,
    ext: &str,
    cases: &[PathBuf],
    verbose: bool,
) -> Result<()> {
    let source = std::fs::read_to_string(solution)?;
    let methods = match ext {
        "cpp" => parser::parse_cpp_signatures(&source)?,
        "rs" => parser::parse_rs_signatures(&source)?,
        _ => anyhow::bail!("unsupported extension: {}", ext),
    };

    let mut all_stats: Vec<MethodStats> = vec![];

    for sig in &methods {
        println!("━━━ {} ━━━", sig.method_name.cyan().bold());

        if verbose {
            println!(
                "{} {}({}) -> {}",
                "signature:".dimmed(),
                sig.method_name,
                sig.params
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect::<Vec<_>>()
                    .join(", "),
                sig.return_type
            );
        }

        let mut stats = MethodStats {
            name: sig.method_name.clone(),
            passed: 0,
            failed: 0,
            total_time_us: 0,
            max_memory_kb: 0,
            time_samples: 0,
        };

        for (i, case_path) in cases.iter().enumerate() {
            let case_num = i + 1;
            let test_data = parser::parse_test_file(case_path)?;

            if verbose {
                println!("{} {:?}", "input:".dimmed(), test_data.inputs);
                println!("{} {}", "expected:".dimmed(), test_data.expected);
            }

            let result = match ext {
                "cpp" => runner::run_cpp(root, solution, sig, &test_data, verbose)?,
                "rs" => runner::run_rs(root, solution, sig, &test_data, verbose)?,
                _ => unreachable!(),
            };

            let actual = result.output.trim();
            let expected = test_data.expected.trim();

            let time_str = result
                .time_us
                .map(|t| format_time(t))
                .unwrap_or_else(|| "?".to_string());
            let mem_str = result
                .memory_kb
                .map(|m| format_memory(m))
                .unwrap_or_else(|| "?".to_string());

            if let Some(t) = result.time_us {
                stats.total_time_us += t;
                stats.time_samples += 1;
            }
            if let Some(m) = result.memory_kb {
                stats.max_memory_kb = stats.max_memory_kb.max(m);
            }

            if parser::outputs_equal(actual, expected) {
                println!(
                    "  {} {}  {}  {}",
                    format!("test {}:", case_num).green(),
                    "passed".green(),
                    time_str.dimmed(),
                    mem_str.dimmed()
                );
                stats.passed += 1;
            } else {
                println!(
                    "  {} {}  {}  {}",
                    format!("test {}:", case_num).red(),
                    "failed".red(),
                    time_str.dimmed(),
                    mem_str.dimmed()
                );
                println!("    {} {}", "expected:".dimmed(), expected);
                println!("    {} {}", "actual:".dimmed(), actual);
                stats.failed += 1;
            }
        }

        all_stats.push(stats);
        println!();
    }

    println!("{}", "summary".bold());
    for stats in &all_stats {
        let total = stats.passed + stats.failed;
        let avg_time = if stats.time_samples > 0 {
            format_time(stats.total_time_us / stats.time_samples as u64)
        } else {
            "?".to_string()
        };
        let status = if stats.failed == 0 {
            format!("{}/{} passed", stats.passed, total).green()
        } else {
            format!("{}/{} passed", stats.passed, total).yellow()
        };
        println!(
            "  {}: {}  avg {}  max {}",
            stats.name.cyan(),
            status,
            avg_time,
            format_memory(stats.max_memory_kb)
        );
    }

    Ok(())
}

fn generate_input_for_type(ty: &str, size: usize) -> String {
    let base = ty.trim_end_matches('&').trim_end_matches('*').trim();
    match base {
        "int" | "long long" => size.to_string(),
        "string" => format!("\"{}\"", "a".repeat(size)),
        t if t.starts_with("vector<vector<") => {
            let inner = (0..size.min(20))
                .map(|_| {
                    let row: Vec<String> = (0..size).map(|_| (rand_int() % 100).to_string()).collect();
                    format!("[{}]", row.join(","))
                })
                .collect::<Vec<_>>();
            format!("[{}]", inner.join(","))
        }
        t if t.starts_with("vector<") => {
            let vals: Vec<String> = (0..size).map(|_| (rand_int() % 100).to_string()).collect();
            format!("[{}]", vals.join(","))
        }
        _ => size.to_string(),
    }
}

fn rand_int() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish()
}

fn estimate_complexity(ratios: &[f64]) -> &'static str {
    if ratios.is_empty() {
        return "?";
    }
    let avg: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
    match avg {
        r if r < 1.3 => "O(1)",
        r if r < 2.5 => "O(n)",
        r if r < 5.0 => "O(n²)",
        r if r < 10.0 => "O(n³)",
        r if r < 20.0 => "O(n⁴)",
        _ => "O(2ⁿ)",
    }
}

fn analyze_solution(
    root: &Path,
    solution: &Path,
    ext: &str,
    sizes: &[usize],
    runs: usize,
    verbose: bool,
) -> Result<()> {
    let source = std::fs::read_to_string(solution)?;
    let methods = match ext {
        "cpp" => parser::parse_cpp_signatures(&source)?,
        "rs" => parser::parse_rs_signatures(&source)?,
        _ => anyhow::bail!("unsupported extension: {}", ext),
    };

    for sig in &methods {
        println!("━━━ {} ━━━", sig.method_name.cyan().bold());

        let mut timings: Vec<(usize, u64)> = vec![];

        for &size in sizes {
            let inputs: Vec<String> = sig
                .params
                .iter()
                .map(|(_, ty)| generate_input_for_type(ty, size))
                .collect();

            let test_data = parser::TestData {
                inputs,
                expected: String::new(),
            };

            let mut times: Vec<u64> = vec![];
            for _ in 0..runs {
                let result = match ext {
                    "cpp" => runner::run_cpp(root, solution, sig, &test_data, false)?,
                    "rs" => runner::run_rs(root, solution, sig, &test_data, false)?,
                    _ => unreachable!(),
                };
                if let Some(t) = result.time_us {
                    times.push(t);
                }
            }

            if !times.is_empty() {
                times.sort();
                let median = times[times.len() / 2];
                timings.push((size, median));

                if verbose {
                    println!(
                        "  n={:<5} {}  (runs: {:?})",
                        size,
                        format_time(median),
                        times.iter().map(|&t| format_time(t)).collect::<Vec<_>>()
                    );
                } else {
                    println!("  n={:<5} {}", size, format_time(median));
                }
            }
        }

        let ratios: Vec<f64> = timings
            .windows(2)
            .filter_map(|w| {
                let (_n1, t1) = w[0];
                let (_n2, t2) = w[1];
                if t1 > 0 {
                    Some(t2 as f64 / t1 as f64)
                } else {
                    None
                }
            })
            .collect();

        let growth_ratios: Vec<String> = timings
            .windows(2)
            .map(|w| {
                let t1 = w[0].1 as f64;
                let t2 = w[1].1 as f64;
                if t1 > 0.0 {
                    format!("{:.1}x", t2 / t1)
                } else {
                    "?".to_string()
                }
            })
            .collect();

        println!();
        println!("  {} {}", "growth ratios:".dimmed(), growth_ratios.join(" → "));
        println!(
            "  {} {}",
            "estimated:".dimmed(),
            estimate_complexity(&ratios).green().bold()
        );
        println!();
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = find_lc_root(cli.directory)?;

    match cli.command {
        Commands::Run {
            file,
            test,
            verbose,
            analyze,
            sizes,
            runs,
        } => {
            let (solution, ext) = resolve_solution(&root, &file)?;
            let problem = solution
                .file_stem()
                .and_then(|s| s.to_str())
                .context("invalid filename")?;

            let mut cases = find_test_cases(&root, problem)?;
            if cases.is_empty() {
                anyhow::bail!("no test cases found for problem {}", problem);
            }

            if let Some(t) = test {
                if t == 0 || t > cases.len() {
                    anyhow::bail!("test {} out of range (1-{})", t, cases.len());
                }
                cases = vec![cases[t - 1].clone()];
            }

            println!(
                "{} {} ({} test{})",
                "running".cyan().bold(),
                solution.file_name().unwrap().to_str().unwrap(),
                cases.len(),
                if cases.len() == 1 { "" } else { "s" }
            );
            println!();

            run_tests(&root, &solution, &ext, &cases, verbose)?;

            if analyze {
                let sizes: Vec<usize> = sizes
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();

                println!();
                println!(
                    "{} (sizes: {:?}, {} runs each)",
                    "complexity analysis".cyan().bold(),
                    sizes,
                    runs
                );
                println!();

                analyze_solution(&root, &solution, &ext, &sizes, runs, verbose)?;
            }
        }
        Commands::List => {
            let mut problems: Vec<String> = std::fs::read_dir(&root)?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_str()?.to_string();
                    if name.ends_with(".cpp") || name.ends_with(".rs") {
                        Some(name)
                    } else {
                        None
                    }
                })
                .collect();
            problems.sort();
            for p in problems {
                println!("{}", p);
            }
        }
    }

    Ok(())
}
