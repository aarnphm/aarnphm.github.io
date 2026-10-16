import type { IncomingMessage, ServerResponse } from 'node:http'
import { spawn } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { constants as fsConstants } from 'node:fs'
import { access, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

const MAX_CODE_LENGTH = 100_000
const MAX_IMPORTS_LENGTH = 256
const LOCAL_TIMEOUT_MS = 30_000
const DEFAULT_CACHE_DIR = path.resolve(process.cwd(), 'quartz', '.quartz-cache', 'lean-verify')

export interface LeanPosition {
  line: number
  column: number
}

export interface LeanDiagnosticMessage {
  severity: 'error' | 'warning' | 'info'
  pos?: LeanPosition
  endPos?: LeanPosition
  data: string
  kind?: string
}

export interface LeanSorryDiagnostic {
  pos?: LeanPosition
  endPos?: LeanPosition
  goal?: string
  goals?: string[]
  message?: string
}

export interface LeanVerifyPayload {
  messages: LeanDiagnosticMessage[]
  sorries: LeanSorryDiagnostic[]
}

export interface LeanLocalVerifyOptions {
  cacheDir?: string
  leanBin?: string
  origin?: string
  token?: string
  timeoutMs?: number
}

export interface LeanLocalVerifyResult {
  status: number
  payload: LeanVerifyPayload | { error: string }
  cache: 'HIT' | 'MISS' | 'BYPASS'
  sha256?: string
}

async function isExecutable(filePath: string): Promise<boolean> {
  try {
    await access(filePath, fsConstants.X_OK)
    return true
  } catch {
    return false
  }
}

let cachedLeanBinary: string | null | undefined

export async function resolveLeanBinary(explicit?: string): Promise<string | null> {
  const override = explicit?.trim() || process.env.LEAN_BIN?.trim()
  if (override) {
    return (await isExecutable(override)) ? override : null
  }
  if (cachedLeanBinary !== undefined) return cachedLeanBinary

  const home = os.homedir()
  if (home) {
    const toolchainsDir = path.join(home, '.elan', 'toolchains')
    try {
      const entries = await readdir(toolchainsDir, { withFileTypes: true })
      const dirs = entries
        .filter(
          entry =>
            entry.isDirectory() && !entry.name.endsWith('.tmp') && !entry.name.endsWith('.lock'),
        )
        .map(entry => entry.name)
        .sort()
        .reverse()
      for (const dir of dirs) {
        const candidate = path.join(toolchainsDir, dir, 'bin', 'lean')
        if (await isExecutable(candidate)) {
          cachedLeanBinary = candidate
          return candidate
        }
      }
    } catch {
      // Ignore missing ~/.elan/toolchains
    }

    const elanProxy = path.join(home, '.elan', 'bin', 'lean')
    if (await isExecutable(elanProxy)) {
      cachedLeanBinary = elanProxy
      return elanProxy
    }
  }

  const pathEnv = process.env.PATH ?? ''
  for (const segment of pathEnv.split(path.delimiter)) {
    if (!segment) continue
    const candidate = path.join(segment, 'lean')
    if (await isExecutable(candidate)) {
      cachedLeanBinary = candidate
      return candidate
    }
  }

  cachedLeanBinary = null
  return null
}

function normalizeSeverity(value: unknown): 'error' | 'warning' | 'info' {
  if (value === 'error' || value === 1) return 'error'
  if (value === 'warning' || value === 2) return 'warning'
  return 'info'
}

function readPos(value: unknown): LeanPosition | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  const record = value as Record<string, unknown>
  const line = typeof record.line === 'number' ? record.line : undefined
  const column = typeof record.column === 'number' ? record.column : 0
  return line !== undefined ? { line, column } : undefined
}

const LOCAL_MATHLIB_PRELUDE = [
  'universe u v w',
  'macro "Type*" : term => `(Type _)',
  'def Set (α : Type u) := α → Prop',
  'namespace Set',
  'variable {α : Type u} {β : Type v}',
  'protected def Mem (s : Set α) (a : α) : Prop := s a',
  'instance : Membership α (Set α) := ⟨Set.Mem⟩',
  'def setOf (p : α → Prop) : Set α := p',
  'macro "{" x:ident " : " t:term " | " p:term "}" : term => `(Set.setOf (fun ($x : $t) => $p))',
  'macro "{" x:ident " | " p:term "}" : term => `(Set.setOf (fun $x => $p))',
  '@[ext] theorem ext {a b : Set α} (h : ∀ x, x ∈ a ↔ x ∈ b) : a = b := funext (fun x => propext (h x))',
  'protected def Subset (a b : Set α) : Prop := ∀ ⦃x⦄, x ∈ a → x ∈ b',
  'instance : HasSubset (Set α) := ⟨Set.Subset⟩',
  'instance : EmptyCollection (Set α) := ⟨fun _ => False⟩',
  'def univ : Set α := fun _ => True',
  'instance : Union (Set α) := ⟨fun a b x => x ∈ a ∨ x ∈ b⟩',
  'instance : Inter (Set α) := ⟨fun a b x => x ∈ a ∧ x ∈ b⟩',
  'instance : SDiff (Set α) := ⟨fun a b x => x ∈ a ∧ x ∉ b⟩',
  'def compl (s : Set α) : Set α := fun x => x ∉ s',
  'postfix:max "ᶜ" => Set.compl',
  '@[simp] theorem mem_setOf {x : α} {p : α → Prop} : x ∈ Set.setOf p ↔ p x := Iff.rfl',
  '@[simp] theorem mem_union (x : α) (a b : Set α) : x ∈ a ∪ b ↔ x ∈ a ∨ x ∈ b := Iff.rfl',
  '@[simp] theorem mem_inter_iff (x : α) (a b : Set α) : x ∈ a ∩ b ↔ x ∈ a ∧ x ∈ b := Iff.rfl',
  '@[simp] theorem mem_diff (x : α) (a b : Set α) : x ∈ a \\ b ↔ x ∈ a ∧ x ∉ b := Iff.rfl',
  '@[simp] theorem mem_compl_iff (s : Set α) (x : α) : x ∈ sᶜ ↔ x ∉ s := Iff.rfl',
  '@[simp] theorem mem_empty_iff_false (x : α) : x ∈ (∅ : Set α) ↔ False := Iff.rfl',
  '@[simp] theorem mem_univ (x : α) : x ∈ (univ : Set α) ↔ True := Iff.rfl',
  'theorem diff_eq (s t : Set α) : s \\ t = s ∩ tᶜ := rfl',
  'def preimage (f : α → β) (s : Set β) : Set α := fun x => f x ∈ s',
  'infixl:80 " ⁻¹\' " => Set.preimage',
  "@[simp] theorem mem_preimage {f : α → β} {s : Set β} {x : α} : x ∈ f ⁻¹' s ↔ f x ∈ s := Iff.rfl",
  'def image (f : α → β) (s : Set α) : Set β := fun y => ∃ x, x ∈ s ∧ f x = y',
  'infixl:80 " \'\' " => Set.image',
  "@[simp] theorem mem_image (f : α → β) (s : Set α) (y : β) : y ∈ f '' s ↔ ∃ x, x ∈ s ∧ f x = y := Iff.rfl",
  'end Set',
].join('\n')

const LOCAL_MATHLIB_PRELUDE_HASH = createHash('sha256')
  .update(LOCAL_MATHLIB_PRELUDE)
  .digest('hex')
  .slice(0, 12)

const preludeBuildPromises = new Map<string, Promise<string | null>>()

async function ensurePrecompiledMathlibPrelude(
  leanBin: string,
  cacheDir: string,
  timeoutMs: number,
): Promise<string | null> {
  const key = `${leanBin}\0${cacheDir}\0${LOCAL_MATHLIB_PRELUDE_HASH}`
  const existing = preludeBuildPromises.get(key)
  if (existing) return existing

  const promise = (async () => {
    const preludeDir = path.join(cacheDir, `prelude-${LOCAL_MATHLIB_PRELUDE_HASH}`)
    const oleanPath = path.join(preludeDir, 'Mathlib.olean')
    try {
      await access(oleanPath, fsConstants.R_OK)
      return preludeDir
    } catch {
      // Build Mathlib.olean below
    }

    try {
      await mkdir(preludeDir, { recursive: true })
      const leanSourcePath = path.join(preludeDir, 'Mathlib.lean')
      await writeFile(leanSourcePath, `${LOCAL_MATHLIB_PRELUDE}\n`, 'utf8')
      const built = await new Promise<boolean>(resolve => {
        const child = spawn(leanBin, ['-R', preludeDir, '-o', oleanPath, leanSourcePath], {
          stdio: ['ignore', 'ignore', 'ignore'],
        })
        const timer = setTimeout(() => {
          child.kill('SIGKILL')
          resolve(false)
        }, timeoutMs)
        child.on('error', () => {
          clearTimeout(timer)
          resolve(false)
        })
        child.on('close', code => {
          clearTimeout(timer)
          resolve(code === 0)
        })
      })
      return built ? preludeDir : null
    } catch {
      return null
    }
  })()

  preludeBuildPromises.set(key, promise)
  return promise
}

function findMathlibImportLine(source: string): number | undefined {
  const lines = source.split('\n')
  for (let index = 0; index < lines.length; index++) {
    if (/^\s*import\s+Mathlib(?:\.[A-Za-z0-9_']+)*\s*(?:--.*)?$/.test(lines[index])) {
      return index + 1
    }
  }
  return undefined
}

function elideMathlibImports(source: string): { code: string; elidedLine?: number } {
  const lines = source.split('\n')
  let elidedLine: number | undefined
  const rewritten = lines.map((line, index) => {
    if (/^\s*import\s+Mathlib(?:\.[A-Za-z0-9_']+)*\s*(?:--.*)?$/.test(line)) {
      if (elidedLine === undefined) elidedLine = index + 1
      return `-- ${line}`
    }
    return line
  })
  return { code: rewritten.join('\n'), elidedLine }
}

async function runLeanJsonFile(
  leanBin: string,
  code: string,
  timeoutMs: number,
  leanPath?: string,
): Promise<{ stdout: string; stderr: string; timedOut: boolean }> {
  const tmpFile = path.join(os.tmpdir(), `quartz-lean-${randomUUID()}.lean`)
  await writeFile(tmpFile, code, 'utf8')
  try {
    return await new Promise(resolve => {
      const env = leanPath
        ? {
            ...process.env,
            LEAN_PATH: process.env.LEAN_PATH
              ? `${process.env.LEAN_PATH}${path.delimiter}${leanPath}`
              : leanPath,
          }
        : process.env
      const child = spawn(leanBin, ['--json', tmpFile], { stdio: ['ignore', 'pipe', 'pipe'], env })
      const outChunks: Buffer[] = []
      const errChunks: Buffer[] = []
      let timedOut = false
      const timer = setTimeout(() => {
        timedOut = true
        child.kill('SIGKILL')
      }, timeoutMs)

      child.stdout.on('data', (chunk: Buffer) => outChunks.push(chunk))
      child.stderr.on('data', (chunk: Buffer) => errChunks.push(chunk))
      child.on('error', err => {
        clearTimeout(timer)
        resolve({ stdout: '', stderr: err.message, timedOut })
      })
      child.on('close', () => {
        clearTimeout(timer)
        resolve({
          stdout: Buffer.concat(outChunks).toString('utf8'),
          stderr: Buffer.concat(errChunks).toString('utf8'),
          timedOut,
        })
      })
    })
  } finally {
    await rm(tmpFile, { force: true }).catch(() => {})
  }
}

export function parseLeanJsonLines(stdout: string, elidedMathlibLine?: number): LeanVerifyPayload {
  const messages: LeanDiagnosticMessage[] = []
  const sorries: LeanSorryDiagnostic[] = []

  if (elidedMathlibLine !== undefined) {
    messages.push({
      severity: 'info',
      pos: { line: elidedMathlibLine, column: 0 },
      endPos: { line: elidedMathlibLine, column: 0 },
      data: 'import Mathlib resolved via local core Lean 4 prelude',
    })
  }

  for (const rawLine of stdout.split(/\r?\n/)) {
    const trimmed = rawLine.trim()
    if (!trimmed.startsWith('{')) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(trimmed)
    } catch {
      continue
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) continue
    const record = parsed as Record<string, unknown>
    const data =
      typeof record.data === 'string'
        ? record.data
        : typeof record.message === 'string'
          ? record.message
          : typeof record.text === 'string'
            ? record.text
            : undefined
    if (data === undefined) continue
    const pos = readPos(record.pos ?? record.position)
    const endPos = readPos(record.endPos ?? record.endPosition)
    const kind = typeof record.kind === 'string' ? record.kind : undefined
    const severity = normalizeSeverity(record.severity)

    messages.push({
      severity,
      ...(pos ? { pos } : {}),
      ...(endPos ? { endPos } : {}),
      data,
      ...(kind ? { kind } : {}),
    })

    if (typeof record.goal === 'string' || Array.isArray(record.goals)) {
      sorries.push({
        ...(pos ? { pos } : {}),
        ...(endPos ? { endPos } : {}),
        ...(typeof record.goal === 'string' ? { goal: record.goal } : {}),
        ...(Array.isArray(record.goals)
          ? { goals: record.goals.filter((g): g is string => typeof g === 'string') }
          : {}),
      })
    }
  }

  return { messages, sorries }
}

async function readDiskCache(cacheDir: string, sha256: string): Promise<LeanVerifyPayload | null> {
  const filePath = path.join(cacheDir, `${sha256}.json`)
  try {
    const raw = await readFile(filePath, 'utf8')
    const parsed = JSON.parse(raw) as Record<string, unknown>
    if (Array.isArray(parsed.messages) && Array.isArray(parsed.sorries)) {
      return {
        messages: parsed.messages as LeanDiagnosticMessage[],
        sorries: parsed.sorries as LeanSorryDiagnostic[],
      }
    }
  } catch {
    // Cache miss
  }
  return null
}

async function writeDiskCache(
  cacheDir: string,
  sha256: string,
  payload: LeanVerifyPayload,
): Promise<void> {
  try {
    await mkdir(cacheDir, { recursive: true })
    await writeFile(path.join(cacheDir, `${sha256}.json`), JSON.stringify(payload), 'utf8')
  } catch {
    // Ignore cache write errors
  }
}

export async function verifyLeanCodeLocal(
  code: string,
  imports = 'mathlib',
  options: LeanLocalVerifyOptions = {},
): Promise<LeanLocalVerifyResult> {
  if (!code) {
    return { status: 400, payload: { error: 'code required' }, cache: 'BYPASS' }
  }
  if (code.length > MAX_CODE_LENGTH) {
    return { status: 413, payload: { error: 'code too large' }, cache: 'BYPASS' }
  }
  const normalizedImports =
    typeof imports === 'string' && imports.length <= MAX_IMPORTS_LENGTH ? imports : 'mathlib'
  const sha256 = createHash('sha256').update(`${normalizedImports}\0${code}`).digest('hex')
  const cacheDir = options.cacheDir ?? DEFAULT_CACHE_DIR

  const cached = await readDiskCache(cacheDir, sha256)
  if (cached) {
    return { status: 200, payload: cached, cache: 'HIT', sha256 }
  }

  const origin = (options.origin ?? process.env.LEAN_VERIFY_ORIGIN)?.trim()
  const timeoutMs = options.timeoutMs ?? LOCAL_TIMEOUT_MS
  if (origin) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const token = (options.token ?? process.env.LEAN_VERIFY_TOKEN)?.trim()
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      }
      if (token) headers.Authorization = `Bearer ${token}`
      const upstream = await fetch(origin, {
        method: 'POST',
        headers,
        body: JSON.stringify({ code, imports: normalizedImports }),
        signal: controller.signal,
      })
      if (!upstream.ok) {
        return {
          status: 502,
          payload: { error: `upstream error: ${upstream.status}` },
          cache: 'BYPASS',
          sha256,
        }
      }
      const json = (await upstream.json()) as LeanVerifyPayload
      if (Array.isArray(json.messages) || Array.isArray(json.sorries)) {
        const normalized: LeanVerifyPayload = {
          messages: Array.isArray(json.messages) ? json.messages : [],
          sorries: Array.isArray(json.sorries) ? json.sorries : [],
        }
        await writeDiskCache(cacheDir, sha256, normalized)
        return { status: 200, payload: normalized, cache: 'MISS', sha256 }
      }
      return { status: 200, payload: json, cache: 'MISS', sha256 }
    } catch (error) {
      const aborted = error instanceof Error && error.name === 'AbortError'
      return {
        status: 504,
        payload: { error: aborted ? 'verification timed out' : 'verification upstream error' },
        cache: 'BYPASS',
        sha256,
      }
    } finally {
      clearTimeout(timer)
    }
  }

  const leanBin = await resolveLeanBinary(options.leanBin)
  if (!leanBin) {
    return {
      status: 503,
      payload: { error: 'lean verification is not configured' },
      cache: 'BYPASS',
      sha256,
    }
  }

  const hasUserLeanPath = Boolean(process.env.LEAN_PATH?.trim())
  const mathlibLine = findMathlibImportLine(code)
  let preludeDir: string | null = null
  if (!hasUserLeanPath && mathlibLine !== undefined) {
    preludeDir = await ensurePrecompiledMathlibPrelude(leanBin, cacheDir, timeoutMs)
  }

  const initialPrepared =
    hasUserLeanPath || preludeDir !== null || mathlibLine === undefined
      ? { code, elidedLine: mathlibLine }
      : elideMathlibImports(code)

  let runResult = await runLeanJsonFile(
    leanBin,
    initialPrepared.code,
    timeoutMs,
    preludeDir ?? undefined,
  )
  let elidedLine = initialPrepared.elidedLine

  if (
    (hasUserLeanPath || preludeDir !== null) &&
    /unknown package 'Mathlib'|module Mathlib\b.*does not exist|unknown module prefix 'Mathlib'/i.test(
      `${runResult.stdout}\n${runResult.stderr}`,
    )
  ) {
    const fallback = elideMathlibImports(code)
    if (fallback.elidedLine !== undefined) {
      elidedLine = fallback.elidedLine
      runResult = await runLeanJsonFile(leanBin, fallback.code, timeoutMs)
    }
  }

  if (runResult.timedOut) {
    return { status: 504, payload: { error: 'verification timed out' }, cache: 'BYPASS', sha256 }
  }

  const payload = parseLeanJsonLines(runResult.stdout, elidedLine)
  await writeDiskCache(cacheDir, sha256, payload)
  return { status: 200, payload, cache: 'MISS', sha256 }
}

export async function handleLocalLeanVerifyNodeRequest(
  req: IncomingMessage,
  res: ServerResponse,
  options: LeanLocalVerifyOptions = {},
): Promise<void> {
  const originHeader = req.headers.origin
  const corsHeaders: Record<string, string> = originHeader
    ? {
        'Access-Control-Allow-Origin': originHeader,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        Vary: 'Origin',
      }
    : {}

  if (req.method === 'OPTIONS') {
    res.writeHead(204, corsHeaders)
    res.end()
    return
  }

  if (req.method !== 'POST') {
    res.writeHead(405, {
      'Content-Type': 'application/json',
      Allow: 'POST, OPTIONS',
      ...corsHeaders,
    })
    res.end(JSON.stringify({ error: 'method not allowed' }))
    return
  }

  const chunks: Buffer[] = []
  let totalBytes = 0
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array)
    totalBytes += buf.length
    if (totalBytes > MAX_CODE_LENGTH * 2) {
      res.writeHead(413, { 'Content-Type': 'application/json', ...corsHeaders })
      res.end(JSON.stringify({ error: 'code too large' }))
      return
    }
    chunks.push(buf)
  }

  let body: { code?: unknown; imports?: unknown }
  try {
    body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as {
      code?: unknown
      imports?: unknown
    }
  } catch {
    res.writeHead(400, { 'Content-Type': 'application/json', ...corsHeaders })
    res.end(JSON.stringify({ error: 'invalid json' }))
    return
  }

  const code = typeof body.code === 'string' ? body.code : ''
  const imports = typeof body.imports === 'string' ? body.imports : 'mathlib'
  const result = await verifyLeanCodeLocal(code, imports, options)

  res.writeHead(result.status, {
    'Content-Type': 'application/json',
    'Cache-Control': result.status === 200 ? 'public, max-age=86400' : 'no-store',
    'X-Lean-Verify-Cache': result.cache,
    ...(result.sha256 ? { 'X-Lean-Verify-Sha256': result.sha256 } : {}),
    ...corsHeaders,
  })
  res.end(JSON.stringify(result.payload))
}
