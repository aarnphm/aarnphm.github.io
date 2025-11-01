import { spawnSync, type SpawnSyncOptions } from "node:child_process"
import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"

function run(command: string, args: string[], opts: SpawnSyncOptions = {}) {
  const res = spawnSync(command, args, { stdio: "inherit", ...opts })
  if (res.status !== 0) {
    const cmd = [command, ...args].join(" ")
    throw new Error(`Command failed (${res.status}): ${cmd}`)
  }
}

function tryRun(command: string, args: string[], opts: SpawnSyncOptions = {}) {
  const res = spawnSync(command, args, { stdio: "inherit", ...opts })
  return res.status === 0
}

function parseMdLfsPatterns(gitattributesPath: string): string[] {
  if (!existsSync(gitattributesPath)) return []
  const lines = readFileSync(gitattributesPath, "utf8").split(/\r?\n/)
  const patterns: string[] = []

  for (let raw of lines) {
    // strip trailing comments
    const hash = raw.indexOf("#")
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trim()
    if (!line) continue
    if (line.startsWith("[")) continue // attribute macros, ignore

    const parts = line.match(/\S+/g) ?? []
    if (parts.length === 0) continue
    const pattern = parts[0]
    const attrs = parts.slice(1).join(" ")

    // Only consider patterns that look like they target Markdown files
    const isMarkdown = /(^|\/)\*?\*?[^\s]*\.md(x)?(\b|$)/i.test(pattern)
    // Only consider entries that reference LFS in any attribute
    const mentionsLfs = /(^|\s)(filter=lfs|diff=lfs|merge=lfs|lfs)(\s|$)/i.test(attrs)

    if (isMarkdown && mentionsLfs) {
      patterns.push(pattern)
    }
  }
  return patterns
}

async function main() {
  const cwd = process.cwd()
  const gitattributesPath = resolve(cwd, ".gitattributes")
  const patterns = parseMdLfsPatterns(gitattributesPath)

  if (patterns.length > 0) {
    const hasGit = tryRun("git", ["--version"])
    const hasLfs = hasGit && tryRun("git", ["lfs", "version"])

    if (!hasGit) {
      console.warn("git not found; skipping LFS checkout.")
    } else if (!hasLfs) {
      console.warn("git lfs not available; skipping LFS checkout.")
    } else {
      console.log("Fetching LFS objects for Markdown patterns from .gitattributes...")
      // Ensure the local repo has LFS enabled (no-op if already configured)
      tryRun("git", ["lfs", "install", "--local"])
      // Fetch only the required objects for these patterns to avoid 'content not local'
      const include = patterns.join(",")
      // Explicit empty exclude to override any global excludes
      run("git", ["lfs", "fetch", "--include", include, "--exclude", ""])
      console.log("Checking out LFS-tracked Markdown files...")
      // Checkout only the matched patterns
      run("git", ["lfs", "checkout", "--", ...patterns])
    }
  } else {
    console.log("No LFS-tracked *.md patterns found in .gitattributes; skipping checkout.")
  }

  console.log("Starting Quartz build...")
  run("pnpm", [
    "exec",
    "quartz/bootstrap-cli.mjs",
    "build",
    "--concurrency",
    "4",
    "--bundleInfo",
    "--verbose",
  ])
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err))
  process.exit(1)
})
