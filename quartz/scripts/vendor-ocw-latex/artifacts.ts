import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import { gunzipSync } from 'node:zlib'
import type { ConvertedSection } from './converter-model'
import { LatexGateError } from './parser'
import { stripComments, type Preamble } from './preamble'

export const run = promisify(execFile)

// ---------------------------------------------------------------------------
// aux, toc, synctex

export function parseAux(text: string): Map<string, { value: string; page: number }> {
  const out = new Map<string, { value: string; page: number }>()
  for (const match of text.matchAll(/\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{(\d+)\}/g)) {
    if (match[1].startsWith('sub@')) continue
    out.set(match[1], { value: match[2].replace(/\\relax\s*/g, '').trim(), page: Number(match[3]) })
  }
  return out
}

export function parseToc(text: string): Map<number, number> {
  const out = new Map<number, number>()
  for (const match of text.matchAll(
    /\\contentsline \{section\}\{\\tocsection \{\}\{(\d+)\}\{.*\}\}\{(\d+)\}/g,
  )) {
    out.set(Number(match[1]), Number(match[2]))
  }
  if (out.size === 0) {
    for (const match of text.matchAll(
      /\\contentsline \{section\}\{\\numberline \{(\d+)\}.*\}\{(\d+)\}/g,
    )) {
      out.set(Number(match[1]), Number(match[2]))
    }
  }
  return out
}

/** Page → first source line (input 1) from a synctex file. */
export function parseSynctex(text: string): { page: number; firstLine: number }[] {
  const pages: { page: number; firstLine: number }[] = []
  let current: { page: number; firstLine: number } | null = null
  const content = text.slice(text.indexOf('\nContent:'))
  for (const line of content.split('\n')) {
    if (line.startsWith('{')) {
      current = { page: Number(line.slice(1)), firstLine: Number.MAX_SAFE_INTEGER }
      continue
    }
    if (line.startsWith('}')) {
      if (current) pages.push(current)
      current = null
      continue
    }
    if (!current) continue
    const match = /^[[(hvxkg$]1,(\d+):/.exec(line)
    if (match) current.firstLine = Math.min(current.firstLine, Number(match[1]))
  }
  return pages
    .filter(page => page.firstLine !== Number.MAX_SAFE_INTEGER)
    .sort((a, b) => a.page - b.page)
}

export function pageOfLine(
  pages: { page: number; firstLine: number }[],
  line: number,
): number | null {
  let result: number | null = null
  for (const page of pages) {
    if (page.firstLine <= line) result = page.page
    else break
  }
  return result
}

// ---------------------------------------------------------------------------
// filesystem pass

async function exists(file: string): Promise<boolean> {
  try {
    await fs.access(file)
    return true
  } catch {
    return false
  }
}

export async function which(tool: string): Promise<boolean> {
  try {
    await run('which', [tool])
    return true
  } catch {
    return false
  }
}

export interface ZipSources {
  dir: string
  aux: Map<string, { value: string; page: number }> | null
  toc: Map<number, number> | null
  synctex: { page: number; firstLine: number }[] | null
  lineOffset: number
  base: string
}

export async function extractZip(zipPath: string, notesSource: string): Promise<ZipSources | null> {
  if (!(await which('unzip'))) return null
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ocw-latex-'))
  await run('unzip', ['-o', '-q', zipPath, '-x', '__MACOSX/*', '-d', dir])
  const texFiles = await walk(dir, name => name.endsWith('.tex'))
  const notesBody = stripLeadingComments(notesSource)
  for (const file of texFiles) {
    const text = await fs.readFile(file, 'utf8')
    if (stripLeadingComments(text) !== notesBody) continue
    const base = path.basename(file, '.tex')
    const sourceDir = path.dirname(file)
    const lineOffset = leadingCommentLines(notesSource) - leadingCommentLines(text)
    const read = async (ext: string) => {
      const candidate = path.join(sourceDir, `${base}${ext}`)
      return (await exists(candidate)) ? fs.readFile(candidate) : null
    }
    const aux = await read('.aux')
    const toc = await read('.toc')
    const synctex = await read('.synctex.gz')
    return {
      dir: sourceDir,
      base,
      lineOffset,
      aux: aux ? parseAux(aux.toString('utf8')) : null,
      toc: toc ? parseToc(toc.toString('utf8')) : null,
      synctex: synctex ? parseSynctex(gunzipSync(synctex).toString('utf8')) : null,
    }
  }
  return null
}

function leadingCommentLines(text: string): number {
  let count = 0
  for (const line of text.split('\n')) {
    if (line.startsWith('%') || line.trim() === '') count++
    else break
  }
  return count
}

function stripLeadingComments(text: string): string {
  return text.split('\n').slice(leadingCommentLines(text)).join('\n').replace(/\s+$/, '')
}

export async function walk(dir: string, accept: (name: string) => boolean): Promise<string[]> {
  const out: string[] = []
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...(await walk(full, accept)))
    else if (accept(entry.name)) out.push(full)
  }
  return out.sort()
}

export interface FigureJob {
  name: string
  kind: 'input' | 'inline' | 'graphics'
  body: string
  line: number
  section: ConvertedSection
}

export async function compileFigures(
  jobs: FigureJob[],
  zip: ZipSources | null,
  preamble: Preamble,
  outDir: string,
  report: string[],
): Promise<Map<string, string | null>> {
  const results = new Map<string, string | null>()
  const haveTools = (await which('latex')) && (await which('dvisvgm'))
  const work = await fs.mkdtemp(path.join(os.tmpdir(), 'ocw-tikz-'))
  // latex and dvisvgm run inside `work`, so every output path must be absolute
  outDir = path.resolve(outDir)
  const preambleLines = [
    '\\documentclass[border=2pt]{standalone}',
    '\\usepackage{amssymb}',
    '\\usepackage{amsmath}',
    '\\usepackage{tikz}',
    '\\usepackage{pgfplots}',
    '\\usetikzlibrary{calc}',
    '\\usetikzlibrary{patterns}',
    ...preamble.colors,
    ...Array.from(
      preamble.operators,
      ([name, body]) => `\\DeclareMathOperator{\\${name}}{${body}}`,
    ),
    ...Array.from(preamble.macros, ([name, body]) => `\\newcommand{\\${name}}{${body}}`),
  ]
  const unique = new Map<string, FigureJob>()
  for (const job of jobs) if (!unique.has(job.name)) unique.set(job.name, job)
  const queue = Array.from(unique.values())
  let cursor = 0
  const worker = async () => {
    while (cursor < queue.length) {
      const job = queue[cursor++]
      const outName = `${job.name.replace(/\s+/g, '-')}`
      if (job.kind === 'graphics') {
        const source = zip ? await findFile(zip.dir, job.name) : null
        if (!source) {
          report.push(`image ${job.name} not in the zip (line ${job.line}); placeholder written`)
          results.set(job.name, null)
          continue
        }
        const target = path.join(outDir, `${outName}`)
        await fs.mkdir(outDir, { recursive: true })
        await fs.copyFile(source, target)
        results.set(job.name, path.basename(target))
        continue
      }
      const svgName = `${outName}.svg`
      const target = path.join(outDir, svgName)
      if (!haveTools) {
        if (await exists(target)) results.set(job.name, svgName)
        else throw new LatexGateError(`latex/dvisvgm missing and ${svgName} is not committed`)
        continue
      }
      let bodyLine: string
      if (job.kind === 'input') {
        const source = zip ? await findFile(zip.dir, `${job.name}.tex`) : null
        if (!source) {
          report.push(`\\input{${job.name}} not in the zip (line ${job.line}); placeholder written`)
          results.set(job.name, null)
          continue
        }
        const body = await fs.readFile(source, 'utf8')
        if (
          !/\\(draw|fill|node|path|addplot|shade|clip|pgf|includegraphics)/.test(
            stripComments(body),
          )
        ) {
          report.push(`\\input{${job.name}} draws nothing (line ${job.line}); placeholder written`)
          results.set(job.name, null)
          continue
        }
        bodyLine = `\\input{${job.name}}`
      } else {
        if (!/\\(draw|fill|node|path|addplot|shade|clip|pgf)/.test(stripComments(job.body))) {
          report.push(`inline tikzpicture at line ${job.line} draws nothing; placeholder written`)
          results.set(job.name, null)
          continue
        }
        await fs.writeFile(path.join(work, `${job.name}.tex`), job.body)
        bodyLine = `\\input{${job.name}}`
      }
      const wrapper = `wrap-${outName}`
      await fs.writeFile(
        path.join(work, `${wrapper}.tex`),
        [...preambleLines, '\\begin{document}', bodyLine, '\\end{document}', ''].join('\n'),
      )
      const texInputs = `.:${zip ? `${zip.dir}//:` : ''}`
      try {
        await run(
          'latex',
          ['-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', `${wrapper}.tex`],
          { cwd: work, env: { ...process.env, TEXINPUTS: texInputs }, maxBuffer: 64 * 1024 * 1024 },
        )
        await run('dvisvgm', ['--no-fonts', '--exact-bbox', '-o', target, `${wrapper}.dvi`], {
          cwd: work,
          maxBuffer: 64 * 1024 * 1024,
        })
        await fs.mkdir(outDir, { recursive: true })
        const svg = await fs.readFile(target, 'utf8')
        // dvisvgm stamps nothing, but strip any comment so reruns stay byte-identical
        await fs.writeFile(target, svg.replace(/<!--[\s\S]*?-->\n?/g, ''))
        results.set(job.name, svgName)
      } catch (error) {
        const log = await fs.readFile(path.join(work, `${wrapper}.log`), 'utf8').catch(() => '')
        const reason =
          /^! (.*)$/m.exec(log)?.[1] ?? (error instanceof Error ? error.message : String(error))
        report.push(
          `figure ${job.name} failed to compile (line ${job.line}): ${reason}; placeholder written`,
        )
        results.set(job.name, null)
      }
    }
  }
  await Promise.all(Array.from({ length: 4 }, worker))
  await fs.rm(work, { recursive: true, force: true })
  return results
}

async function findFile(dir: string, name: string): Promise<string | null> {
  const direct = path.join(dir, name)
  if (await exists(direct)) return direct
  const files = await walk(dir, () => true)
  const base = path.basename(name).toLowerCase()
  return files.find(file => path.basename(file).toLowerCase() === base) ?? null
}
