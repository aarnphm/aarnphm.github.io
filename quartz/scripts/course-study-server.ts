import yaml from 'js-yaml'
/**
 * Dev-only study endpoint. The course pages built by the watcher post here to record study state
 * in authored files, which stay in the checkout for aarnphm to commit:
 *
 * - `POST /study`    `lectures/index.md` `study.read` / `study.problems` dates
 * - `POST /promote`  a new `lectures/<nn>.md` + `lectures/<nn>.fc` (never overwrites)
 * - `POST /pset`     `psets/<nn>.md`, seeded with the set's problems when missing or a bare stub, and
 *                    its `problems:` states
 *
 * It listens on 127.0.0.1 only and answers localhost origins only. The deployed site never sees
 * it: production builds render no endpoint (`courseStudyEndpoint`).
 */
import { access, mkdir, readFile, writeFile } from 'node:fs/promises'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import path from 'node:path'
import { parseStudyLog, psetNumber } from '../util/course'
import {
  COURSE_DIR,
  STUDY_DATE,
  parseProblems,
  type ProblemRecord,
  type ProblemState,
} from '../util/course-study'

const MAX_BODY = 512 * 1024
const pad = (n: number) => String(n).padStart(2, '0')

class StudyError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
  }
}

interface Frontmatter {
  head: string
  body: string
  data: Record<string, unknown>
}

function splitFrontmatter(source: string): Frontmatter {
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(source)
  if (!match) return { head: '', body: source, data: {} }
  const loaded = yaml.load(match[1])
  const data =
    typeof loaded === 'object' && loaded !== null ? (loaded as Record<string, unknown>) : {}
  return { head: match[1], body: source.slice(match[0].length), data }
}

/**
 * Replace one top-level key of a frontmatter block with `block` (its full `key:` lines), leaving
 * every other line as written. Appends the block when the key is absent.
 */
function setTopLevel(head: string, key: string, block: string): string {
  const lines = head.split('\n')
  const start = lines.findIndex(line => line.startsWith(`${key}:`))
  if (start < 0) return [...lines, block].join('\n')
  let end = start + 1
  while (end < lines.length && (lines[end] === '' || /^\s/.test(lines[end]))) end++
  return [...lines.slice(0, start), block, ...lines.slice(end)].join('\n')
}

const quote = (value: string) => `'${value.replaceAll("'", "''")}'`

function dateMapBlock(indent: string, key: string, map: Map<number, string>): string {
  if (map.size === 0) return `${indent}${key}: {}`
  const rows = [...map.entries()].sort(([a], [b]) => a - b)
  return [`${indent}${key}:`, ...rows.map(([n, date]) => `${indent}  ${n}: ${quote(date)}`)].join(
    '\n',
  )
}

function problemsBlock(map: Map<number, ProblemRecord>): string {
  if (map.size === 0) return 'problems: {}'
  const rows = [...map.entries()].sort(([a], [b]) => a - b)
  return [
    'problems:',
    ...rows.map(([n, record]) => {
      const fields = [`state: ${record.state}`]
      if (record.date) fields.push(`date: ${quote(record.date)}`)
      if (record.reveal) fields.push(`reveal: ${quote(record.reveal)}`)
      return `  ${n}: { ${fields.join(', ')} }`
    }),
  ].join('\n')
}

function assemble(head: string, body: string): string {
  return `---\n${head.replace(/\n+$/, '')}\n---\n${body}`
}

/** One set in `psets/problems.json`, the course's transcription of the OCW problem set PDFs. */
interface SetText {
  n: number
  due?: string | null
  reading?: string | null
  intro?: string | null
  /** Term of the official solution OCW ships, when it is another term's set. */
  solution?: string | null
  problems: { n: number; title: string; statement: string }[]
}

const quoted = (text: string) =>
  text
    .trim()
    .split('\n')
    .map(line => (line ? `> ${line}` : '>'))
    .join('\n')

/** One `## N. title` section per problem: the statement, its `^problem-N` block id, an attempt. */
function problemSection(n: number, title: string, statement: string): string {
  return [
    `## ${n}. ${title}`,
    '',
    `> [!question] problem ${n}`,
    quoted(statement),
    '',
    `^problem-${n}`,
    '',
    '### attempt',
  ].join('\n')
}

/**
 * The body a new `psets/<nn>.md` starts with: the transcribed set when the course has one, else
 * the problems of the generated `notes/pset-<nn>` page, else the set's resource link alone.
 */
async function psetSeed(
  dir: string,
  name: string,
  n: number,
  resource: string | undefined,
): Promise<{ body: string; solutionTerm?: string }> {
  const sets = await readFile(path.join(dir, 'psets', 'problems.json'), 'utf8')
    .then(text => (JSON.parse(text) as { sets?: SetText[] }).sets ?? [])
    .catch(() => [] as SetText[])
  const set = sets.find(entry => entry.n === n)
  if (set && set.problems.length > 0) {
    const lead = [
      set.due ? `Due ${set.due.replace(/\.$/, '')}.` : '',
      set.reading ? `Reading: ${set.reading.replace(/\.$/, '')}.` : '',
    ].filter(Boolean)
    const parts = [
      ...(lead.length > 0 ? [lead.join(' ')] : []),
      ...(set.intro ? [set.intro.trim()] : []),
      ...set.problems.map(problem => problemSection(problem.n, problem.title, problem.statement)),
    ]
    return {
      body: `\n${parts.join('\n\n').replace(/\n+$/, '')}\n`,
      ...(set.solution && set.solution !== 'none' ? { solutionTerm: set.solution } : {}),
    }
  }
  // a LaTeX course: the generated page interleaves problems with background, so the section quotes
  // each lead (up to its `^problem-N` block id) and links the page for the rest
  const text = await readFile(path.join(dir, 'notes', `pset-${pad(n)}.md`), 'utf8').catch(() => '')
  const leads = Array.from(
    text.matchAll(/^\*\*problem (\d+)\.\*\*[ \t]*([\s\S]*?)^\^problem-\1[ \t]*$/gim),
  )
  if (leads.length > 0) {
    const page = `courses/${name}/notes/pset-${pad(n)}`
    const sections = leads.map(([, m, lead]) => {
      const first = lead.trim().split('\n')[0]
      let title =
        /^\(([^)]{2,80})\)/.exec(first)?.[1] ??
        /^(.{1,70}?[.?!:])(\s|$)/.exec(first)?.[1] ??
        `problem ${m}`
      title = title.replace(/[.:]$/, '')
      const statement = `${lead.trim()}\n\n[[${page}#^problem-${m}|the full statement]]`
      return problemSection(Number(m), title, statement)
    })
    return { body: `\n${sections.join('\n\n')}\n` }
  }
  return {
    body: resource
      ? `\nThe set: [[courses/${name}/resources/${resource}|problem set ${n}]]. Add one \`## <n>. title\` section per problem as you work it.\n`
      : `\nAdd one \`## <n>. title\` section per problem as you work it.\n`,
  }
}

async function exists(file: string): Promise<boolean> {
  try {
    await access(file)
    return true
  } catch {
    return false
  }
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    size += (chunk as Buffer).length
    if (size > MAX_BODY) throw new StudyError(413, 'request too large')
    chunks.push(chunk as Buffer)
  }
  let value: unknown
  try {
    value = JSON.parse(Buffer.concat(chunks).toString('utf8'))
  } catch {
    throw new StudyError(400, 'body is not json')
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new StudyError(400, 'body is not an object')
  }
  return value as Record<string, unknown>
}

function unitNumber(value: unknown): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > 999) {
    throw new StudyError(422, 'n must be an integer between 1 and 999')
  }
  return value
}

function dateOrNull(value: unknown, field: string): string | null {
  if (value === null || value === undefined) return null
  if (typeof value !== 'string' || !STUDY_DATE.test(value)) {
    throw new StudyError(422, `${field} must be YYYY-MM-DD`)
  }
  return value
}

export function startCourseStudyServer(options: {
  root: string
  port: number
  log: (message: string, error?: boolean) => void
}): Server {
  const contentRoot = path.join(options.root, 'content', 'courses')

  /** A vendored course directory: its home carries `layout: course`. */
  async function course(dirValue: unknown) {
    if (typeof dirValue !== 'string' || !COURSE_DIR.test(dirValue) || dirValue.includes('..')) {
      throw new StudyError(422, 'unknown course directory')
    }
    const dir = path.join(contentRoot, dirValue)
    let home: Frontmatter
    try {
      home = splitFrontmatter(await readFile(path.join(dir, 'index.md'), 'utf8'))
    } catch {
      throw new StudyError(404, 'unknown course directory')
    }
    const record = home.data.course as Record<string, unknown> | undefined
    if (home.data.layout !== 'course' || typeof record !== 'object' || record === null) {
      throw new StudyError(404, `${dirValue} has no course block`)
    }
    return { name: dirValue, dir, record }
  }

  const relative = (file: string) => path.relative(options.root, file)

  async function study(input: Record<string, unknown>) {
    const { name, dir, record } = await course(input.dir)
    const key = input.key
    if (key !== 'read' && key !== 'problems') throw new StudyError(422, 'key is read or problems')
    const n = unitNumber(input.n)
    const date = dateOrNull(input.date, 'date')
    const file = path.join(dir, 'lectures', 'index.md')
    const source = (await exists(file))
      ? await readFile(file, 'utf8')
      : assemble(
          [
            `title: ${quote(`${String(record.number ?? name)} study log`)}`,
            `description: ${quote(`the days each unit of ${String(record.number ?? name)} was read and its problems worked`)}`,
            'layout: study-log',
          ].join('\n'),
          '',
        )
    const fm = splitFrontmatter(source)
    const log = parseStudyLog(fm.data)
    if (date) log[key].set(n, date)
    else log[key].delete(n)
    const block = [
      'study:',
      dateMapBlock('  ', 'read', log.read),
      dateMapBlock('  ', 'problems', log.problems),
    ].join('\n')
    await mkdir(path.dirname(file), { recursive: true })
    await writeFile(file, assemble(setTopLevel(fm.head, 'study', block), fm.body))
    options.log(`study: ${name} ${key} ${n} → ${date ?? 'cleared'}`)
    return { path: relative(file), key, n, date }
  }

  async function promote(input: Record<string, unknown>) {
    const { name, dir } = await course(input.dir)
    const n = unitNumber(input.n)
    const { md, fc } = input
    if (typeof md !== 'string' || !md.startsWith('---\n')) {
      throw new StudyError(422, 'md must be a markdown file with frontmatter')
    }
    if (typeof fc !== 'string') throw new StudyError(422, 'fc must be a string')
    const mdFile = path.join(dir, 'lectures', `${pad(n)}.md`)
    const fcFile = path.join(dir, 'lectures', `${pad(n)}.fc`)
    for (const file of [mdFile, fcFile]) {
      if (await exists(file)) throw new StudyError(409, `${relative(file)} exists; nothing written`)
    }
    await mkdir(path.dirname(mdFile), { recursive: true })
    await writeFile(mdFile, md.endsWith('\n') ? md : `${md}\n`, { flag: 'wx' })
    const written = [relative(mdFile)]
    if (fc.trim()) {
      await writeFile(fcFile, fc.endsWith('\n') ? fc : `${fc}\n`, { flag: 'wx' })
      written.push(relative(fcFile))
    }
    options.log(`promote: ${name} ${pad(n)} → ${written.join(', ')}`)
    return { written }
  }

  async function pset(input: Record<string, unknown>) {
    const { name, dir, record } = await course(input.dir)
    const n = unitNumber(input.n)
    const file = path.join(dir, 'psets', `${pad(n)}.md`)
    const created = !(await exists(file))
    // a stub from before seeding (no problem sections, no states) is seeded in place
    const stub = created ? undefined : splitFrontmatter(await readFile(file, 'utf8'))
    const bare =
      stub !== undefined && !/^## /m.test(stub.body) && parseProblems(stub.data).size === 0
    if (created || bare) {
      // the problem set's resource page, from the vendored calendar's `due` ids
      const units = Array.isArray(record.units) ? (record.units as Record<string, unknown>[]) : []
      const id = units
        .flatMap(unit => (Array.isArray(unit.due) ? (unit.due as unknown[]) : []))
        .find((due): due is string => typeof due === 'string' && psetNumber(due) === n)
      const seed = await psetSeed(dir, name, n, id)
      const number = String(record.number ?? name)
      let head =
        stub?.head ??
        [
          `title: ${quote(`${number} problem set ${pad(n)}`)}`,
          `description: ${quote(`attempts and solutions for ${number} problem set ${pad(n)}`)}`,
          'tags:',
          '  - course',
          'problems: {}',
        ].join('\n')
      if (seed.solutionTerm)
        head = setTopLevel(head, 'solutionTerm', `solutionTerm: ${quote(seed.solutionTerm)}`)
      await mkdir(path.dirname(file), { recursive: true })
      await writeFile(file, assemble(head, seed.body), created ? { flag: 'wx' } : undefined)
      options.log(`pset: ${created ? 'created' : 'seeded'} ${relative(file)}`)
    }
    if (input.problem === undefined) return { path: relative(file), created }
    const problem = unitNumber(input.problem)
    const fm = splitFrontmatter(await readFile(file, 'utf8'))
    const problems = parseProblems(fm.data)
    const current: ProblemRecord = problems.get(problem) ?? { state: 'open' }
    const day = dateOrNull(input.date, 'date')
    const next: ProblemRecord = { ...current }
    if (input.state !== undefined) {
      const state = input.state as ProblemState
      if (state !== 'open' && state !== 'stuck' && state !== 'solved') {
        throw new StudyError(422, 'state is open, stuck or solved')
      }
      next.state = state
      if (day) next.date = day
    }
    const reveal = dateOrNull(input.reveal, 'reveal')
    if (reveal) next.reveal = reveal
    if (next.state === 'open' && !next.reveal) problems.delete(problem)
    else problems.set(problem, next)
    await writeFile(
      file,
      assemble(setTopLevel(fm.head, 'problems', problemsBlock(problems)), fm.body),
    )
    options.log(`pset: ${name} ${pad(n)} #${problem} → ${next.state}${reveal ? ' (revealed)' : ''}`)
    return { path: relative(file), created, problem, record: problems.get(problem) ?? null }
  }

  const routes: Record<string, (input: Record<string, unknown>) => Promise<unknown>> = {
    '/study': study,
    '/promote': promote,
    '/pset': pset,
  }

  const allowedOrigin = (origin: string | undefined) => {
    if (!origin) return null
    try {
      const url = new URL(origin)
      return url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname)
        ? origin
        : null
    } catch {
      return null
    }
  }

  const send = (res: ServerResponse, origin: string | null, status: number, body: unknown) => {
    res.writeHead(status, {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      ...(origin ? { 'Access-Control-Allow-Origin': origin, Vary: 'Origin' } : {}),
    })
    res.end(JSON.stringify(body))
  }

  const server = createServer((req, res) => {
    const origin = allowedOrigin(req.headers.origin)
    if (req.headers.origin && !origin) return send(res, null, 403, { error: 'origin not allowed' })
    const pathname = new URL(req.url ?? '/', 'http://localhost').pathname
    const route = Object.hasOwn(routes, pathname) ? routes[pathname] : undefined
    if (req.method === 'OPTIONS') {
      res.writeHead(
        route ? 204 : 404,
        origin
          ? {
              'Access-Control-Allow-Origin': origin,
              'Access-Control-Allow-Methods': 'POST',
              'Access-Control-Allow-Headers': 'Content-Type',
              'Access-Control-Max-Age': '600',
              Vary: 'Origin',
            }
          : {},
      )
      return res.end()
    }
    if (!route) return send(res, origin, 404, { error: 'unknown route' })
    if (req.method !== 'POST') return send(res, origin, 405, { error: 'use POST' })
    readJson(req)
      .then(route)
      .then(result => send(res, origin, 200, { ok: true, ...(result as object) }))
      .catch(error => {
        const status = error instanceof StudyError ? error.status : 500
        if (status === 500) options.log(`study endpoint failed: ${String(error)}`, true)
        send(res, origin, status, { error: error instanceof Error ? error.message : 'failed' })
      })
  })
  server.on('error', error => options.log(`study endpoint unavailable: ${error.message}`, true))
  server.listen(options.port, '127.0.0.1', () =>
    options.log(`study endpoint on http://localhost:${options.port}`),
  )
  return server
}
