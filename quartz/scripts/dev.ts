// orchestrates quartz dev rebuilds with wrangler dev resets when public/ is regenerated.
import { spawn, ChildProcessWithoutNullStreams } from "node:child_process"
import { access } from "node:fs/promises"
import path from "node:path"
import process from "node:process"
import { fileURLToPath } from "node:url"

const gitRoot = resolveGitRoot()
const publicDir = path.join(gitRoot, "public")
const pollIntervalMs = 500
const useColor = process.stdout.isTTY && process.stderr.isTTY
const RESET = "\x1b[0m"
const labelNames = {
  main: "main",
  "pnpm:dev": "pnpm:dev",
  wrangler: "wrangler",
} as const
type Label = keyof typeof labelNames
type ManagedChild = ChildProcessWithoutNullStreams & { label: Label }
const formattedLabels: Record<Label, string> = formatLabels(labelNames)
let shuttingDown = false
let lifecycleRegistered = false
let pnpmDev: ManagedChild | null = null
let wrangler: ManagedChild | null = null
let rawInputEnabled = false

void main()

function resolveGitRoot(): string {
  const envRoot = process.env.GIT_ROOT
  if (envRoot && envRoot.length > 0) {
    return path.resolve(envRoot)
  }
  const current = path.dirname(fileURLToPath(import.meta.url))
  return path.resolve(current, "..", "..")
}

async function main(): Promise<void> {
  pnpmDev = startProcess(["dev"], "pnpm:dev")
  registerLifecycle()
  registerCtrlCHandler()
  pnpmDev.on("exit", (code, signal) => {
    if (!shuttingDown) {
      log(
        "main",
        `pnpm dev exited with code ${code ?? "null"} signal ${signal ?? "null"}`,
        "stderr",
      )
      void shutdown(1)
    }
  })
  pnpmDev.on("error", (err) => {
    log("main", `failed to launch pnpm dev: ${describeError(err)}`, "stderr")
    void shutdown(1)
  })
  await manageWranglerLoop()
}

function registerLifecycle(): void {
  if (lifecycleRegistered) return
  lifecycleRegistered = true
  const signals: NodeJS.Signals[] = ["SIGINT", "SIGTERM", "SIGHUP"]
  for (const sig of signals) {
    process.on(sig, () => {
      void shutdown(0, sig)
    })
  }
  process.on("uncaughtException", (err) => {
    log("main", `manager encountered error: ${describeError(err)}`, "stderr")
    void shutdown(1)
  })
  process.on("unhandledRejection", (reason) => {
    log("main", `manager unhandled rejection: ${describeError(reason)}`, "stderr")
    void shutdown(1)
  })
}

async function manageWranglerLoop(): Promise<void> {
  while (!shuttingDown) {
    const exists = await pathExists(publicDir)
    if (exists && wrangler === null) {
      wrangler = startProcess(["dlx", "wrangler", "dev"], "wrangler")
      wrangler.on("exit", (code, signal) => {
        log(
          "main",
          `wrangler exited with code ${code ?? "null"} signal ${signal ?? "null"}`,
          code === 0 && signal === null ? "stdout" : "stderr",
        )
        wrangler = null
      })
      wrangler.on("error", (err) => {
        log("main", `failed to launch wrangler dev: ${describeError(err)}`, "stderr")
        wrangler = null
      })
    }
    if (!exists && wrangler) {
      log("main", "stopping wrangler while public directory rebuilds")
      await stopProcess(wrangler)
      wrangler = null
    }
    await delay(pollIntervalMs)
  }
}

async function shutdown(code: number, signal?: NodeJS.Signals): Promise<void> {
  if (shuttingDown) return
  shuttingDown = true
  if (signal) {
    log("main", `received ${signal}, shutting down...`)
  }
  if (rawInputEnabled && process.stdin.isTTY) {
    process.stdin.setRawMode(false)
    process.stdin.pause()
    process.stdin.removeListener("data", handleStdin)
    rawInputEnabled = false
  }
  if (wrangler) {
    await stopProcess(wrangler)
    wrangler = null
  }
  if (pnpmDev) {
    await stopProcess(pnpmDev)
    pnpmDev = null
  }
  process.exit(code)
}

function startProcess(args: string[], label: Label): ManagedChild {
  log("main", `starting ${label}`)
  // @ts-ignore
  const proc = spawn("pnpm", args, {
    cwd: gitRoot,
    env: {
      ...process.env,
      ...(label === "wrangler"
        ? { PUBLIC_BASE_URL: process.env.PUBLIC_BASE_URL ?? "http://localhost:8080" }
        : {}),
    },
    stdio: ["ignore", "pipe", "pipe"],
  }) as ManagedChild
  proc.label = label
  pipeStream(proc)
  return proc
}

async function stopProcess(child: ManagedChild): Promise<void> {
  if (child.exitCode !== null || child.signalCode) {
    return
  }
  const exitPromise = once(child, "exit")
  child.kill("SIGINT")
  const first = await Promise.race([
    exitPromise.then(() => "exit" as const),
    delay(3000).then(() => "timeout" as const),
  ])
  if (first === "timeout" && child.exitCode === null) {
    child.kill("SIGTERM")
    const second = await Promise.race([
      exitPromise.then(() => "exit" as const),
      delay(2000).then(() => "timeout" as const),
    ])
    if (second === "timeout" && child.exitCode === null) {
      child.kill("SIGKILL")
      await exitPromise
    }
  }
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await access(target)
    return true
  } catch {
    return false
  }
}

function pipeStream(child: ManagedChild): void {
  child.stdout.on("data", (chunk) => {
    process.stdout.write(formatChunk(child.label, chunk))
  })
  child.stderr.on("data", (chunk) => {
    process.stderr.write(formatChunk(child.label, chunk))
  })
}

function formatChunk(label: Label, chunk: Buffer): string {
  const lines = chunk.toString()
  const prefix = formattedLabels[label] ?? `[${label}]`
  return lines
    .split(/(?<=\n)/)
    .map((line) => (line.length > 0 ? `${prefix} ${line}` : line))
    .join("")
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function registerCtrlCHandler(): void {
  if (!process.stdin.isTTY || rawInputEnabled) return
  process.stdin.setRawMode(true)
  process.stdin.resume()
  process.stdin.on("data", handleStdin)
  rawInputEnabled = true
}

function handleStdin(data: Buffer): void {
  for (const byte of data) {
    if (byte === 0x03) {
      void shutdown(0, "SIGINT")
      return
    }
  }
}

function once(child: ManagedChild, event: "exit"): Promise<void> {
  return new Promise((resolve) => {
    child.once(event, () => resolve())
  })
}

function log(label: Label, message: string, stream: "stdout" | "stderr" = "stdout"): void {
  const prefix = formattedLabels[label] ?? `[${label}]`
  const target = stream === "stdout" ? process.stdout : process.stderr
  target.write(`${prefix} ${message}\n`)
}

function formatLabels<T extends Record<string, string>>(map: T): Record<keyof T, string> {
  const widest = Math.max(...Object.values(map).map((value) => value.length))
  const result = {} as Record<keyof T, string>
  for (const [key, name] of Object.entries(map) as Array<[keyof T, string]>) {
    const padded = name.padEnd(widest, " ")
    const base = `[${padded}]`
    if (!useColor) {
      result[key] = base
      continue
    }
    const color = labelColor(key as Label)
    result[key] = color ? `${color}${base}${RESET}` : base
  }
  return result
}

function describeError(value: unknown): string {
  if (value instanceof Error) {
    return value.stack ?? value.message
  }
  if (typeof value === "string") {
    return value
  }
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function labelColor(label: Label): string | null {
  switch (label) {
    case "main":
      return "\x1b[38;5;39m"
    case "pnpm:dev":
      return "\x1b[38;5;110m"
    case "wrangler":
      return "\x1b[38;5;214m"
    default:
      return null
  }
}
