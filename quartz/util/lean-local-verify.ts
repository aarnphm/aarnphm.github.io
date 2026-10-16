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
): Promise<{ stdout: string; stderr: string; timedOut: boolean }> {
  const tmpFile = path.join(os.tmpdir(), `quartz-lean-${randomUUID()}.lean`)
  await writeFile(tmpFile, code, 'utf8')
  try {
    return await new Promise(resolve => {
      const child = spawn(leanBin, ['--json', tmpFile], { stdio: ['ignore', 'pipe', 'pipe'] })
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
      data: 'import Mathlib elided for local core Lean 4 checking',
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

  const hasLeanPath = Boolean(process.env.LEAN_PATH?.trim())
  const prepared = hasLeanPath ? { code, elidedLine: undefined } : elideMathlibImports(code)
  let runResult = await runLeanJsonFile(leanBin, prepared.code, timeoutMs)
  let elidedLine = prepared.elidedLine

  if (
    hasLeanPath &&
    elidedLine === undefined &&
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
