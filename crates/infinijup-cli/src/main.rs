use anyhow::Result;
use clap::{Parser, Subcommand};
use infinijup::{enforce_no_uvm_ipc, PreflightConfig};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "infinijupyter", version, about = "Infinijupyter CLI")] 
struct Cli {
  #[arg(long, global = true, default_value = "info")] 
  log: String,
  #[command(subcommand)]
  cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
  /// Run preflight checks (no-UVM/IPC enforcement) for a PID
  Preflight { #[arg(long)] pid: i32 },
  /// Launch vLLM under supervision (placeholder)
  Vllm { #[arg(trailing_var_arg = true)] args: Vec<String> },
  /// Launch a Jupyter kernel under supervision (placeholder)
  Kernel { #[arg(trailing_var_arg = true)] args: Vec<String> },
}

fn main() -> Result<()> {
  tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::new(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into())))
    .init();
  let cli = Cli::parse();
  match cli.cmd {
    Cmd::Preflight { pid } => {
      let rep = enforce_no_uvm_ipc(pid, PreflightConfig::default());
      match rep {
        Ok(r) => {
          println!("OK: {:?}", r);
          Ok(())
        }
        Err(e) => {
          eprintln!("Preflight FAILED: {e}");
          std::process::exit(2);
        }
      }
    }
    Cmd::Vllm { args } => {
      println!("[placeholder] would launch vllm with args: {:?}", args);
      Ok(())
    }
    Cmd::Kernel { args } => {
      println!("[placeholder] would launch kernel with args: {:?}", args);
      Ok(())
    }
  }
}

