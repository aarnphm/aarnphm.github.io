import path from "node:path"
import fs from "node:fs/promises"
import { createWriteStream } from "node:fs"
import { Readable } from "node:stream"
import { pipeline as streamPipeline } from "node:stream/promises"

const HF_BASE_URL = "https://huggingface.co"

type HFSibling = {
  rfilename: string
  size?: number
  oid?: string
  lfs?: { size?: number; sha256?: string }
}

type HFModelMeta = {
  sha?: string
  siblings?: HFSibling[]
}

type EnsureOptions = {
  outputDir: string
  modelId: string
  revision?: string
  token?: string
  force?: boolean
  mirrorDirs?: string[]
}

type EnsureResult = {
  downloaded: number
  skipped: number
  revision: string
}

const MODEL_META_FILENAME = ".hf-meta.json"

const SAFE_SEGMENT = /[^a-zA-Z0-9._-]/g

const REQUIRED_SUFFIXES = new Set([
  ".json",
  ".onnx",
  ".txt",
  ".model",
  ".vocab",
  ".tokenizer",
  ".merges",
])

const REQUIRED_PREFIXES = ["onnx/"]

function sanitizeSegment(segment: string): string {
  if (segment === "" || segment === "." || segment === "..") {
    throw new Error(`invalid path segment '${segment}' in model id`)
  }
  return segment.replace(SAFE_SEGMENT, "_")
}

function ensureLeadingSlash(pathname: string): string {
  if (pathname.startsWith("/")) return pathname
  return `/${pathname}`
}

function safeJoin(root: string, relative: string): string {
  const segments = relative.split("/").map((part) => {
    if (part === "" || part === "." || part === "..") {
      throw new Error(`invalid path component '${part}' in model file '${relative}'`)
    }
    return part
  })
  const joined = path.join(root, ...segments)
  if (!joined.startsWith(root)) {
    throw new Error(`attempted path traversal for '${relative}'`)
  }
  return joined
}

function shouldDownloadFile(file: HFSibling): boolean {
  const { rfilename } = file
  if (!rfilename) return false
  for (const prefix of REQUIRED_PREFIXES) {
    if (rfilename.startsWith(prefix)) {
      return true
    }
  }
  const idx = rfilename.lastIndexOf(".")
  if (idx === -1) return false
  const ext = rfilename.slice(idx).toLowerCase()
  return REQUIRED_SUFFIXES.has(ext)
}

async function fetchJSON<T>(url: string, token?: string): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" }
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  const res = await fetch(url, { headers })
  if (!res.ok) {
    throw new Error(`failed to fetch ${url}: ${res.status} ${res.statusText}`)
  }
  return (await res.json()) as T
}

async function downloadFile(url: string, dest: string, token?: string) {
  const headers: Record<string, string> = {}
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  const res = await fetch(url, { headers })
  if (!res.ok || !res.body) {
    throw new Error(`failed to download ${url}: ${res.status} ${res.statusText}`)
  }
  await fs.mkdir(path.dirname(dest), { recursive: true })
  const nodeStream = Readable.fromWeb(res.body as any)
  await streamPipeline(nodeStream, createWriteStream(dest))
}

async function fileMatchesExpected(dest: string, file: HFSibling): Promise<boolean> {
  try {
    const stat = await fs.stat(dest)
    const expected = file.lfs?.size ?? file.size
    if (expected === undefined) return stat.size > 0
    return stat.size === expected
  } catch {
    return false
  }
}

async function mirrorFileIntoRoots(
  src: string,
  file: HFSibling,
  mirrorRoots: string[],
): Promise<void> {
  if (mirrorRoots.length === 0) return
  await Promise.all(
    mirrorRoots.map(async (root) => {
      const dest = safeJoin(root, file.rfilename)
      await fs.mkdir(path.dirname(dest), { recursive: true })
      if (await fileMatchesExpected(dest, file)) {
        return
      }
      await fs.copyFile(src, dest)
    }),
  )
}

export function computeModelLocalPath(modelId: string): string {
  const segments = modelId.split("/").filter(Boolean).map(sanitizeSegment)
  return ensureLeadingSlash(["models", ...segments].join("/"))
}

export async function ensureLocalModel({
  outputDir,
  modelId,
  revision,
  token,
  force,
  mirrorDirs,
}: EnsureOptions): Promise<EnsureResult> {
  if (!modelId) {
    return { downloaded: 0, skipped: 0, revision: "" }
  }
  const modelSegments = modelId.split("/").filter(Boolean).map(sanitizeSegment)
  const modelRoot = path.join(outputDir, "models", ...modelSegments)
  const mirrorRoots = (mirrorDirs ?? [])
    .filter((dir) => dir && dir.trim().length > 0)
    .map((dir) => path.join(dir, ...modelSegments))
  const metaPath = path.join(modelRoot, MODEL_META_FILENAME)
  let previousSha: string | undefined
  if (!force) {
    try {
      const meta = JSON.parse(await fs.readFile(metaPath, "utf-8")) as { sha?: string }
      previousSha = meta.sha
    } catch {}
  }

  const metaUrl = new URL(`/api/models/${modelId}`, HF_BASE_URL)
  metaUrl.searchParams.set("expand", "siblings")
  if (revision) metaUrl.searchParams.set("revision", revision)
  const meta = await fetchJSON<HFModelMeta>(metaUrl.toString(), token)
  const resolvedSha = meta.sha ?? revision ?? "main"
  const siblings = meta.siblings ?? []

  const files = siblings.filter(shouldDownloadFile)
  await fs.mkdir(modelRoot, { recursive: true })
  await Promise.all(mirrorRoots.map((root) => fs.mkdir(root, { recursive: true })))

  let downloaded = 0
  let skipped = 0
  for (const file of files) {
    const dest = safeJoin(modelRoot, file.rfilename)
    const upToDate =
      !force && previousSha === resolvedSha && (await fileMatchesExpected(dest, file))
    if (upToDate) {
      skipped += 1
      await mirrorFileIntoRoots(dest, file, mirrorRoots)
      continue
    }
    const fileUrl = new URL(`/${modelId}/resolve/${resolvedSha}/${file.rfilename}`, HF_BASE_URL)
    await downloadFile(fileUrl.toString(), dest, token)
    downloaded += 1
    await mirrorFileIntoRoots(dest, file, mirrorRoots)
  }

  await fs.writeFile(
    metaPath,
    JSON.stringify({
      sha: resolvedSha,
      downloadedAt: new Date().toISOString(),
      files: files.length,
    }),
  )

  if (mirrorRoots.length > 0) {
    const metaCopies = mirrorRoots.map(async (root) => {
      const target = path.join(root, MODEL_META_FILENAME)
      await fs.mkdir(path.dirname(target), { recursive: true })
      await fs.copyFile(metaPath, target)
    })
    await Promise.all(metaCopies)
  }

  return { downloaded, skipped, revision: resolvedSha }
}
