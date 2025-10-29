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
  "pnpm:exec": "pnpm:exec",
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
let pnpmDevRetriesRemaining = 0
const DEFAULT_DEV_PORT = 7373
const WS_PORT_OFFSET = 1
const WRANGLER_PORT_OFFSET = 707
const MAX_BASE_PORT = 65535 - WRANGLER_PORT_OFFSET

const runtimeConfig = resolveRuntimeConfig(process.argv.slice(2))
const totalPnpmDevAttempts = runtimeConfig.pnpmDevRetryLimit + 1
pnpmDevRetriesRemaining = runtimeConfig.pnpmDevRetryLimit
process.env.PUBLIC_BASE_URL = runtimeConfig.publicBaseUrl

void main()

function resolveGitRoot(): string {
  const envRoot = process.env.GIT_ROOT
  if (envRoot && envRoot.length > 0) {
    return path.resolve(envRoot)
  }
  const current = path.dirname(fileURLToPath(import.meta.url))
  return path.resolve(current, "..", "..")
}

interface RuntimeConfig {
  port: number
  wsPort: number
  wranglerPort: number
  pnpmDevArgs: string[]
  wranglerArgs: string[]
  publicBaseUrl: string
  pnpmDevRetryLimit: number
}

interface CliOptions {
  port: number | null
  help: boolean
  retry: number | null
  force: boolean
}

function assertBasePort(candidate: number): void {
  if (!Number.isInteger(candidate)) {
    failStartup(`port must be an integer between 1 and ${MAX_BASE_PORT}, got ${candidate}`)
  }
  if (candidate < 1 || candidate > MAX_BASE_PORT) {
    failStartup(`port must be between 1 and ${MAX_BASE_PORT}, got ${candidate}`)
  }
}

function resolveRuntimeConfig(argv: string[]): RuntimeConfig {
  const { port, help, retry, force } = parseCliOptions(argv)
  if (help) {
    printHelp()
    process.exit(0)
  }
  const envBaseUrl = process.env.PUBLIC_BASE_URL
  const envPort = parsePortFromBase(envBaseUrl)
  const effectivePort = port ?? envPort ?? DEFAULT_DEV_PORT
  assertBasePort(effectivePort)
  const wsPort = effectivePort + WS_PORT_OFFSET
  const wranglerPort = effectivePort + WRANGLER_PORT_OFFSET
  const publicBaseUrl =
    port !== null
      ? `http://localhost:${effectivePort}`
      : (envBaseUrl ?? `http://localhost:${effectivePort}`)
  const pnpmDevArgs = [
    "exec",
    "quartz/bootstrap-cli.mjs",
    "build",
    "--concurrency",
    "8",
    "--serve",
    "--verbose",
    "--port",
    String(effectivePort),
    "--wsPort",
    String(wsPort),
  ]
  if (force) {
    pnpmDevArgs.push("--force")
  }
  const wranglerArgs = ["dlx", "wrangler", "dev", "--port", String(wranglerPort), "--live-reload"]
  const pnpmDevRetryLimit = retry ?? 2
  return {
    port: effectivePort,
    wsPort,
    wranglerPort,
    pnpmDevArgs,
    wranglerArgs,
    publicBaseUrl,
    pnpmDevRetryLimit,
  }
}

function parseCliOptions(argv: string[]): CliOptions {
  let port: number | null = null
  let help = false
  let force = false
  let retry: number | null = null
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (token === "--") {
      break
    }
    if (token === "--help" || token === "-h") {
      help = true
      continue
    }
    if (token === "--port") {
      const value = argv[index + 1]
      if (!value) {
        failStartup("missing port value for --port")
      }
      index += 1
      port = parsePortValue(value)
      continue
    }
    if (token.startsWith("--port=")) {
      port = parsePortValue(token.slice("--port=".length))
      continue
    }
    if (token === "--retry") {
      const value = argv[index + 1]
      if (!value) {
        failStartup("missing retry value for --retry")
      }
      index += 1
      retry = parseRetryValue(value)
      continue
    }
    if (token.startsWith("--retry=")) {
      retry = parseRetryValue(token.slice("--retry=".length))
    }
    if (token === "--force") {
      force = true
    }
  }
  return { port, help, retry, force }
}

function parsePortValue(raw: string): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) {
    failStartup(`invalid port: ${raw}`)
  }
  assertBasePort(parsed)
  return parsed
}

function parseRetryValue(raw: string): number {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 0) {
    failStartup(`invalid retry count: ${raw}`)
  }
  return parsed
}

function printHelp(): void {
  const prefix = formattedLabels.main ?? "[main]"
  const lines = [
    "usage: pnpm swarm -- [options]",
    "",
    "options:",
    "  --port <port>       bind pnpm dev to <port>, websocket to <port+1>, wrangler dev to <port+2>",
    "  --retry <count>     restart pnpm dev up to <count> times when it exits non-zero",
    "  --force             enforce running all plugins (longer to compile)",
    "  --help, -h          show this message",
    "",
    "environment:",
    "  PUBLIC_BASE_URL     overrides default http://localhost:<port> when --port is not passed",
    "",
    "example:",
    "  pnpm swarm -- --port 8081",
  ]
  for (const line of lines) {
    process.stdout.write(`${prefix} ${line}\n`)
  }
}

function parsePortFromBase(value?: string): number | null {
  if (!value) {
    return null
  }
  try {
    const candidate = value.includes("://") ? value : `http://${value}`
    const url = new URL(candidate)
    if (!url.port) {
      return null
    }
    const parsed = Number(url.port)
    return Number.isInteger(parsed) ? parsed : null
  } catch {
    return null
  }
}

function failStartup(message: string): never {
  const prefix = formattedLabels.main ?? "[main]"
  process.stderr.write(`${prefix} ${message}\n`)
  process.exit(1)
}

async function main(): Promise<void> {
  log(
    "main",
    `using dev port ${runtimeConfig.port}, ws ${runtimeConfig.wsPort}, wrangler ${runtimeConfig.wranglerPort}`,
  )
  log("main", `launching pnpm dev (attempt 1/${totalPnpmDevAttempts})`)
  launchPnpmDev()
  registerLifecycle()
  registerCtrlCHandler()
  await manageWranglerLoop()
}

function launchPnpmDev(): void {
  const attempt = runtimeConfig.pnpmDevRetryLimit - pnpmDevRetriesRemaining + 1
  pnpmDev = startProcess(
    runtimeConfig.pnpmDevArgs,
    "pnpm:exec",
    `starting pnpm:exec (attempt ${attempt}/${totalPnpmDevAttempts})`,
  )
  const child = pnpmDev
  child.on("exit", (code, signal) => handlePnpmDevExit(child, code, signal))
  child.on("error", (err) => handlePnpmDevError(child, err))
}

function handlePnpmDevExit(
  child: ManagedChild,
  code: number | null,
  signal: NodeJS.Signals | null,
): void {
  if (pnpmDev === child) {
    pnpmDev = null
  }
  if (shuttingDown) {
    return
  }
  const stream: "stdout" | "stderr" = code === 0 && signal === null ? "stdout" : "stderr"
  log("main", `pnpm dev exited with code ${code ?? "null"} signal ${signal ?? "null"}`, stream)
  if (code === 0 && signal === null) {
    void shutdown(0)
    return
  }
  retryPnpmDev(`exit code ${code ?? "null"} signal ${signal ?? "null"}`)
}

function handlePnpmDevError(child: ManagedChild, err: Error): void {
  if (pnpmDev === child) {
    pnpmDev = null
  }
  if (shuttingDown) {
    return
  }
  log("main", `pnpm dev error: ${describeError(err)}`, "stderr")
  retryPnpmDev("process error")
}

function retryPnpmDev(reason: string): void {
  if (pnpmDevRetriesRemaining === 0) {
    log(
      "main",
      `pnpm dev restart budget exhausted after ${totalPnpmDevAttempts} attempts`,
      "stderr",
    )
    void shutdown(1)
    return
  }
  pnpmDevRetriesRemaining -= 1
  const attempt = totalPnpmDevAttempts - pnpmDevRetriesRemaining
  log(
    "main",
    `retrying pnpm dev after ${reason} (attempt ${attempt}/${totalPnpmDevAttempts}, ${pnpmDevRetriesRemaining} retries left)`,
    "stderr",
  )
  launchPnpmDev()
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
      wrangler = startProcess(runtimeConfig.wranglerArgs, "wrangler")
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

function startProcess(args: string[], label: Label, message?: string): ManagedChild {
  log("main", message ?? `starting ${label}`)
  // @ts-ignore
  const proc = spawn("pnpm", args, {
    cwd: gitRoot,
    env: {
      ...process.env,
      ...(label === "wrangler" ? { PUBLIC_BASE_URL: runtimeConfig.publicBaseUrl } : {}),
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
    case "pnpm:exec":
      return "\x1b[38;5;110m"
    case "wrangler":
      return "\x1b[38;5;214m"
    default:
      return null
  }
}
