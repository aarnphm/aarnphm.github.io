#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { performance } from 'node:perf_hooks'
import { StringDecoder } from 'node:string_decoder'
import { parseArgs, stripVTControlCharacters } from 'node:util'
import { parseQuartzCacheReceipt } from '../util/cache-receipts.ts'

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    output: { type: 'string' },
    ingest: { type: 'string' },
    'exit-code': { type: 'string' },
    help: { type: 'boolean' },
  },
})
if (values.help || (!values.ingest && positionals.length === 0)) {
  console.log(
    'usage: node quartz/scripts/deploy-log.mjs [--output DIR] -- COMMAND [ARG...]\n       node quartz/scripts/deploy-log.mjs --ingest LOG [--exit-code N] [--output DIR]',
  )
  process.exit(values.help ? 0 : 2)
}
if (values.ingest && positionals.length > 0) throw new Error('Choose a command or --ingest')
const importedExit = values['exit-code'] === undefined ? null : Number(values['exit-code'])
if (
  importedExit !== null &&
  (!Number.isInteger(importedExit) || importedExit < 0 || importedExit > 255)
) {
  throw new Error('--exit-code must be an integer from 0 through 255')
}
if (importedExit !== null && !values.ingest)
  throw new Error('--exit-code is only used with --ingest')

const started = performance.now()
const runId = `${new Date().toISOString().replaceAll(/[:.]/g, '-')}-${process.pid}-${randomUUID().slice(0, 8)}`
const output = path.resolve(
  values.output ?? path.join('docs/verification/deployments/artifacts', runId),
)
// Exclusive creation keeps parallel runs and explicit output paths from overwriting prior logs.
await fs.mkdir(path.dirname(output), { recursive: true })
await fs.mkdir(output, { mode: 0o700 })
const log = createWriteStream(path.join(output, 'deploy.log'), { flags: 'wx', mode: 0o600 })
const run = {
  id: runId,
  cwd: process.cwd(),
  command: positionals,
  sourceLog: values.ingest ? path.resolve(values.ingest) : null,
  startedAt: new Date().toISOString(),
  finishedAt: null,
  status: values.ingest ? 'importing' : 'running',
  exitCode: null,
  signal: null,
  recordingMs: 0,
  elapsedMs: null,
  node: process.version,
  deployStages: [],
  quartzStages: [],
  cacheReceipts: [],
  notebookCacheReceipts: [],
}
let tail = ''
let dirty = false
let pendingSave = Promise.resolve()
let recorderError
let captureFailed = false
const escape = value =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
const seconds = ms => `${(ms / 1000).toFixed(2)} s`
const finiteNumber = value => typeof value === 'number' && Number.isFinite(value)

function parseLine(line) {
  const notebookPrefix = '__QUARTZ_NOTEBOOK_CACHE__'
  const notebookOffset = line.indexOf(notebookPrefix)
  if (notebookOffset >= 0) {
    try {
      const receipt = JSON.parse(line.slice(notebookOffset + notebookPrefix.length))
      if (
        receipt &&
        typeof receipt.epoch === 'string' &&
        typeof receipt.file === 'string' &&
        typeof receipt.reason === 'string' &&
        ['prepare', 'restore'].includes(receipt.phase) &&
        ['candidate', 'hit', 'miss', 'reload'].includes(receipt.status)
      ) {
        const { epoch, file, phase, status, reason } = receipt
        run.notebookCacheReceipts.push({ epoch, file, phase, status, reason })
      }
    } catch {}
    return
  }
  const receipt = parseQuartzCacheReceipt(line)
  if (receipt) {
    run.cacheReceipts.push(receipt)
    return
  }
  for (const prefix of ['__QUARTZ_DEPLOY_STAGE__', '__QUARTZ_BUILD_STAGE__']) {
    const offset = line.indexOf(prefix)
    if (offset < 0) continue
    let payload
    try {
      payload = JSON.parse(line.slice(offset + prefix.length))
    } catch {
      continue
    }
    if (!payload || typeof payload !== 'object' || typeof payload.stage !== 'string') continue
    if (prefix === '__QUARTZ_BUILD_STAGE__') {
      if (!finiteNumber(payload.elapsedMs)) continue
      run.quartzStages.push({
        epoch: typeof payload.epoch === 'string' ? payload.epoch : '',
        stage: payload.stage,
        elapsedMs: payload.elapsedMs,
        atMs: finiteNumber(payload.atMs) ? payload.atMs : null,
        cpu: payload.cpu ?? null,
        elu: payload.elu ?? null,
        memory: payload.memory ?? null,
      })
    } else if (payload.state === 'started') {
      run.deployStages.push({
        name: payload.stage,
        status: 'running',
        startedAt: payload.at ?? null,
        elapsedMs: null,
        exitCode: null,
      })
    } else if (
      payload.state === 'finished' &&
      finiteNumber(payload.exitCode) &&
      finiteNumber(payload.elapsedSeconds)
    ) {
      const stage = run.deployStages.findLast(
        item => item.name === payload.stage && item.status === 'running',
      )
      if (stage)
        Object.assign(stage, {
          status: payload.exitCode === 0 ? 'passed' : 'failed',
          elapsedMs: payload.elapsedSeconds * 1000,
          exitCode: payload.exitCode,
        })
    }
  }
}

function renderReport() {
  const running = run.status === 'running' || run.status === 'importing'
  const deployRows = run.deployStages
    .map(
      stage =>
        `<tr><th scope="row">${escape(stage.name)}</th><td>${escape(stage.status)}${stage.exitCode === null ? '' : ` (${stage.exitCode})`}</td><td>${stage.elapsedMs === null ? '–' : seconds(stage.elapsedMs)}</td></tr>`,
    )
    .join('')
  const quartzRows = run.quartzStages
    .map(
      stage =>
        `<tr><th scope="row">${escape(stage.stage)}<small>${escape(stage.epoch)}</small></th><td>${seconds(stage.elapsedMs)}</td><td>${stage.atMs === null ? '–' : seconds(stage.atMs)}</td></tr>`,
    )
    .join('')
  const cacheRows = run.cacheReceipts
    .map(receipt => {
      const counts = receipt.counts
      const work = counts
        ? `${counts.parsed} parsed · ${counts.rendered} rendered · ${counts.outputWrites} output writes`
        : '–'
      return `<tr><th scope="row">${escape(receipt.phase)} / ${escape(receipt.status)}<small>${escape(receipt.epoch)} · ${escape(receipt.profile)}</small></th><td>${escape(receipt.reason)}<small>${escape(receipt.manifestPath)}${receipt.detail ? `<br>${escape(receipt.detail)}` : ''}</small></td><td>${work}</td></tr>`
    })
    .join('')
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">${running ? '<meta http-equiv="refresh" content="3">' : ''}<title>deployment report · ${escape(run.status)}</title><style>
:root{color-scheme:light dark;--light:#fffcf0;--dark:#100f0f;--darkgray:#6f6e69;--line:#e6e4d9;font-family:"PP Neue Montreal",system-ui,sans-serif;color:var(--dark);background:var(--light)}
@media(prefers-color-scheme:dark){:root{--light:#100f0f;--dark:#cecdc3;--darkgray:#9f9d96;--line:#282726}}
*{box-sizing:border-box}body{margin:0;padding:clamp(16px,4vw,48px)}main{max-width:1100px;margin:auto;min-width:0}h1,h2{font-family:"Space Groteskque",system-ui,sans-serif;font-weight:500}h1{font-size:28px}h2{font-size:20px;margin-top:32px}p,dd{overflow-wrap:anywhere}a{color:inherit;text-underline-offset:3px;display:inline-block;min-height:24px}a:focus-visible,summary:focus-visible{outline:2px solid var(--dark);outline-offset:2px}nav{display:flex;gap:20px;flex-wrap:wrap}small{display:block;color:var(--darkgray);font-size:12px}table{width:100%;border-collapse:collapse;font:13px/1.6 "Berkeley Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid var(--line);overflow-wrap:anywhere}th:first-child{width:52%}table th{font-weight:500}pre,code,dd{font:13px/1.6 "Berkeley Mono",ui-monospace,monospace}pre{padding:16px;border:1px solid var(--line);white-space:pre-wrap;overflow-wrap:anywhere}dt{color:var(--darkgray);font-size:12px;margin-top:12px}dd{margin:0}summary{cursor:pointer;min-height:24px}.note{color:var(--darkgray);font-size:14px}@media(pointer:coarse){a,summary{min-height:44px;padding-top:10px}}@media(max-width:420px){th,td{padding:6px 3px;font-size:12px}}
</style></head><body><main><h1>deployment report</h1><p><strong>${escape(run.status)}</strong>${run.exitCode === null ? ', exit status unknown' : `, exit ${run.exitCode}`}${run.elapsedMs === null ? '' : ` · ${seconds(run.elapsedMs)}`}</p><nav><a href="deploy.log">raw log</a><a href="run.json">timings and metadata</a></nav><dl><dt>${run.sourceLog ? 'imported at' : 'started'}</dt><dd>${escape(run.startedAt)}</dd><dt>${run.sourceLog ? 'ingestion directory' : 'checkout'}</dt><dd>${escape(run.cwd)}</dd><dt>${run.sourceLog ? 'imported from' : 'command'}</dt><dd>${escape(run.sourceLog ?? JSON.stringify(run.command))}</dd></dl>
${deployRows ? `<h2>deployment stages</h2><table><thead><tr><th>stage</th><th>status</th><th>elapsed</th></tr></thead><tbody>${deployRows}</tbody></table>` : ''}
${quartzRows ? `<h2>Quartz build stages</h2><p class="note">Emitter work can overlap. Stage durations cannot be added to obtain total build time. The JSON preserves CPU and event-loop measurements.</p><table><thead><tr><th>stage / epoch</th><th>elapsed</th><th>process clock</th></tr></thead><tbody>${quartzRows}</tbody></table>` : ''}
${cacheRows ? `<h2>cache receipts</h2><p class="note">Zero work is reported when the committed output passes validation. Other decisions leave counts unknown. The raw log retains every receipt.</p><table><thead><tr><th>phase / status</th><th>reason / manifest</th><th>work</th></tr></thead><tbody>${cacheRows}</tbody></table>` : ''}
<h2>log tail</h2><p class="note">The page shows the last 200,000 characters. The raw log keeps the complete output.${running ? ' This page refreshes every three seconds.' : ''}${run.sourceLog ? ' Import time is not deployment time.' : ''}</p><pre>${escape(stripVTControlCharacters(tail))}</pre></main></body></html>`
}

function save() {
  // A failed periodic write must still allow a final snapshot attempt.
  pendingSave = pendingSave
    .catch(() => {})
    .then(async () => {
      run.recordingMs = performance.now() - started
      const json = JSON.stringify(run, null, 2) + '\n'
      const html = renderReport()
      for (const [name, contents] of [
        ['run.json', json],
        ['index.html', html],
      ]) {
        const temporary = path.join(output, `${name}.tmp`)
        await fs.writeFile(temporary, contents, { mode: 0o600 })
        await fs.rename(temporary, path.join(output, name))
      }
      dirty = false
    })
  return pendingSave
}

async function write(stream, chunk) {
  await new Promise((resolve, reject) =>
    stream.write(chunk, error => (error ? reject(error) : resolve())),
  )
}
async function relay(stream, terminal) {
  const decoder = new StringDecoder('utf8')
  let line = ''
  const append = text => {
    tail = (tail + text).slice(-200_000)
    dirty = true
    const lines = (line + text).split('\n')
    line = lines.pop() ?? ''
    for (const complete of lines) parseLine(stripVTControlCharacters(complete))
    // A long unbroken log line must not become an unbounded parser buffer.
    if (line.length > 65_536) line = line.slice(-65_536)
  }
  for await (const chunk of stream) {
    await write(log, chunk)
    if (terminal) {
      try {
        await write(terminal, chunk)
      } catch {
        // A disconnected terminal must not cancel the deployment or its file capture.
        terminal = null
      }
    }
    append(decoder.write(chunk))
  }
  append(decoder.end())
  if (line) parseLine(stripVTControlCharacters(line))
}

let child
let requestedSignal
let killTimer
function interrupt(signal) {
  requestedSignal = signal
  if (!child?.pid) return
  try {
    process.kill(-child.pid, signal)
  } catch (error) {
    if (error.code !== 'ESRCH') throw error
  }
  killTimer ??= setTimeout(() => {
    try {
      process.kill(-child.pid, 'SIGKILL')
    } catch (error) {
      if (error.code !== 'ESRCH') console.error(error)
    }
  }, 10_000).unref()
}
process.on('SIGINT', interrupt)
process.on('SIGTERM', interrupt)
log.on('error', error => {
  recorderError = error
  interrupt('SIGTERM')
})
process.stdout.on('error', () => {})
process.stderr.on('error', () => {})
await save()
console.error(`Deployment report: ${path.join(output, 'index.html')}`)
const interval = setInterval(() => {
  if (dirty)
    void save().catch(error => {
      recorderError = error
      interrupt('SIGTERM')
    })
}, 1000).unref()

try {
  if (values.ingest) {
    await relay(createReadStream(path.resolve(values.ingest)), null)
    run.exitCode = importedExit
    run.status = importedExit === null ? 'imported' : importedExit === 0 ? 'passed' : 'failed'
  } else {
    const [command, ...args] = positionals
    const env = { ...process.env, GARDEN_DEPLOY_LOG_ACTIVE: '1' }
    // Capture pipes hide the terminal's color support from child processes.
    if (
      env.FORCE_COLOR === undefined &&
      env.NO_COLOR === undefined &&
      env.NODE_DISABLE_COLORS === undefined
    ) {
      const terminal = process.stdout.isTTY ? process.stdout : process.stderr
      const depth = terminal.isTTY ? terminal.getColorDepth() : 1
      if (depth > 1) env.FORCE_COLOR = depth >= 24 ? '3' : depth >= 8 ? '2' : '1'
    }
    child = spawn(command, args, {
      cwd: process.cwd(),
      env,
      stdio: ['inherit', 'pipe', 'pipe'],
      detached: true,
    })
    const exited = new Promise((resolve, reject) => {
      child.once('error', reject)
      child.once('exit', (code, signal) => resolve({ code, signal }))
    })
    const [result] = await Promise.all([
      exited,
      relay(child.stdout, process.stdout),
      relay(child.stderr, process.stderr),
    ])
    run.signal = requestedSignal ?? result.signal
    run.exitCode = requestedSignal
      ? requestedSignal === 'SIGINT'
        ? 130
        : 143
      : (result.code ?? (result.signal === 'SIGINT' ? 130 : result.signal === 'SIGTERM' ? 143 : 1))
    run.status = run.signal ? 'interrupted' : run.exitCode === 0 ? 'passed' : 'failed'
    run.elapsedMs = performance.now() - started
  }
} catch (error) {
  captureFailed = true
  run.status = 'failed'
  run.exitCode = error.code === 'ENOENT' ? 127 : 1
  const message = `Deployment recorder: ${error.message}\n`
  tail = (tail + message).slice(-200_000)
  await write(log, message)
  console.error(message.trimEnd())
} finally {
  clearInterval(interval)
  clearTimeout(killTimer)
  process.removeListener('SIGINT', interrupt)
  process.removeListener('SIGTERM', interrupt)
}
run.finishedAt = new Date().toISOString()
for (const stage of run.deployStages) {
  if (stage.status === 'running')
    stage.status = run.status === 'interrupted' ? 'interrupted' : 'incomplete'
}
await new Promise((resolve, reject) => log.end(error => (error ? reject(error) : resolve())))
await save()
if (recorderError) throw recorderError
console.error(`Deployment ${run.status}. Report: ${path.join(output, 'index.html')}`)
process.exitCode = values.ingest && !captureFailed ? 0 : (run.exitCode ?? 1)
