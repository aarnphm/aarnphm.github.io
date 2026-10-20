#!/usr/bin/env node
// Token budget: files over LIMIT tokens, where tokens = bytes / 4.
//   (no flags)       pre-commit: staged blobs; prints cut plans and split steps, exits 1 on violations
//   --all            every tracked file in the index; prints one line per file, exits 1 on violations
//   --paths <p...>   the named files in the index; prints cut plans, exits 1 on violations
//   --json           with any source: a report on stdout for the refactor harness, exits 0
// It reads blobs from the index, so partial staging is measured as it will be committed.
// Skipped: gitlinks, symlinks, binary blobs, and paths that .gitattributes (index copy) marks
// linguist-generated, linguist-vendored or -agent-authored.
import { execFileSync } from 'node:child_process'
import path from 'node:path'

export const LIMIT = 5000
// Size to aim for after a split, so the next few edits do not trip the hook again.
export const TARGET = 4000
export const BYTES_PER_TOKEN = 4
const MAX_BYTES = LIMIT * BYTES_PER_TOKEN
const CUTS_SHOWN = 40

export const tokens = bytes => Math.ceil(bytes / BYTES_PER_TOKEN)
const git = (args, input, env) =>
  execFileSync('git', args, { input, maxBuffer: 1 << 30, env: { ...process.env, ...env } })

const stagedEntries = () => {
  // `:<src mode> <dst mode> <src sha> <dst sha> <status>\0<path>\0` per entry.
  const raw = git(['diff', '--cached', '--raw', '-z', '--no-renames', '--diff-filter=ACMRT'])
    .toString('utf8')
    .split('\0')
  const entries = []
  for (let i = 0; i + 1 < raw.length; i += 2) {
    const [, mode, , sha] = raw[i].split(' ')
    entries.push({ path: raw[i + 1], mode, sha })
  }
  return entries
}

const indexEntries = paths => {
  // `<mode> <sha> <stage>\t<path>\0` per entry.
  const args = ['ls-files', '-s', '-z']
  if (paths) args.push('--', ...paths)
  return git(args, undefined, paths ? { GIT_LITERAL_PATHSPECS: '1' } : {})
    .toString('utf8')
    .split('\0')
    .filter(Boolean)
    .map(line => {
      const [meta, file] = line.split('\t')
      const [mode, sha] = meta.split(' ')
      return { path: file, mode, sha }
    })
}

const isSet = v => v === 'set' || v === 'true'
const isUnset = v => v === 'unset' || v === 'false'

// Over-budget, agent-authored text files among the entries, with their text.
export const overBudget = entries => {
  const files = entries.filter(f => f.mode === '100644' || f.mode === '100755')
  if (files.length === 0) return []
  const sizes = git(['cat-file', '--batch-check=%(objectsize)'], files.map(f => f.sha).join('\n'))
    .toString()
    .trim()
    .split('\n')
  const oversized = files
    .map((f, i) => ({ path: f.path, sha: f.sha, bytes: Number(sizes[i]) }))
    .filter(f => f.bytes > MAX_BYTES)
  if (oversized.length === 0) return []

  const attrs = new Map(oversized.map(f => [f.path, {}]))
  const attrOut = git(
    [
      'check-attr',
      '--cached',
      '-z',
      '--stdin',
      'linguist-generated',
      'linguist-vendored',
      'agent-authored',
      'diff',
    ],
    oversized.map(f => f.path).join('\0'),
  )
    .toString('utf8')
    .split('\0')
  for (let i = 0; i + 2 < attrOut.length; i += 3)
    attrs.get(attrOut[i])[attrOut[i + 1]] = attrOut[i + 2]

  const violations = []
  for (const f of oversized) {
    const a = attrs.get(f.path)
    if (isSet(a['linguist-generated']) || isSet(a['linguist-vendored'])) continue
    if (isUnset(a['agent-authored']) || isUnset(a.diff)) continue
    const blob = git(['cat-file', 'blob', f.sha])
    // Same binary heuristic as git: a NUL byte in the first 8000 bytes.
    if (blob.subarray(0, 8000).includes(0)) continue
    violations.push({ ...f, text: blob.toString('utf8') })
  }
  return violations
}

export const kindOf = p => {
  const base = path.basename(p)
  if (/\.(test|spec)\.[cm]?[jt]sx?$|(^test_.*|_test)\.py$/.test(base)) return 'test'
  if (/\.(s?css|sass|less)$/.test(base)) return 'style'
  if (/\.(md|mdx|markdown)$/.test(base)) return 'docs'
  return 'code'
}

// Top-level blocks: declarations and rules for code and styles, headings for prose.
// Comments and decorators directly above a declaration belong to it.
export const blocks = (text, kind) => {
  const lines = text.split('\n')
  const starts = []
  let fenced = false
  lines.forEach((line, i) => {
    if (kind === 'docs') {
      if (/^(```|~~~)/.test(line)) fenced = !fenced
      if (!fenced && /^#{1,3} /.test(line)) starts.push(i)
    } else if (
      /^[^\s})\]'"`@/*#]|^@(media|container|supports|layer|mixin|function|include|keyframes|property|font-face|each|for|use|forward|import)\b/.test(
        line,
      )
    ) {
      let at = i
      const floor = (starts.at(-1) ?? -1) + 1
      while (at > floor && /^(\/\/|\/\*|\s\*|# |@)/.test(lines[at - 1])) at--
      starts.push(at)
    }
  })
  if (starts[0] !== 0) starts.unshift(0)
  return starts.map((start, j) => {
    const end = (starts[j + 1] ?? lines.length) - 1
    const bytes = Buffer.byteLength(lines.slice(start, end + 1).join('\n')) + 1
    // The first line that is code, not a comment or blank.
    const label = lines.slice(start, end + 1).find(l => l.trim() && !/^\s*(\/\/|\/\*|\*)/.test(l))
    return {
      id: `b${j}`,
      start: start + 1,
      end: end + 1,
      tokens: tokens(bytes),
      label: (label ?? lines[start]).trim().slice(0, 64),
    }
  })
}

// Proposed cuts: consecutive blocks packed up to TARGET tokens, so each part keeps neighbouring
// (usually related) code together and styles keep their cascade order.
export const cuts = blocks => {
  const parts = []
  for (const b of blocks) {
    const last = parts.at(-1)
    if (last && !last.over && last.tokens + b.tokens <= TARGET) {
      last.end = b.end
      last.tokens += b.tokens
      last.blocks.push(b.id)
    } else
      parts.push({
        start: b.start,
        end: b.end,
        tokens: b.tokens,
        label: b.label,
        blocks: [b.id],
        over: b.tokens > TARGET,
      })
  }
  return parts
}

const describe = f => {
  const kind = kindOf(f.path)
  const bs = blocks(f.text, kind)
  const t = tokens(f.bytes)
  return {
    path: f.path,
    blob: f.sha,
    bytes: f.bytes,
    tokens: t,
    kind,
    parts: Math.ceil(t / TARGET),
    blocks: bs,
    cuts: cuts(bs),
  }
}

const cutPlan = d => {
  const out = [
    `${d.path}: ${d.tokens} tokens. Split it into at least ${d.parts} files of at most ${TARGET} tokens each.`,
    '  proposed cuts (line ranges of consecutive top-level blocks):',
  ]
  for (const [i, c] of d.cuts.slice(0, CUTS_SHOWN).entries()) {
    const range = `L${c.start}-${c.end}`.padEnd(14)
    const note = c.over ? '  <- one block over budget: split inside it' : ''
    out.push(
      `    ${String(i + 1).padStart(3)}. ${range}${String(c.tokens).padStart(6)} tok  ${c.label}${note}`,
    )
  }
  if (d.cuts.length > CUTS_SHOWN) out.push(`    ... ${d.cuts.length - CUTS_SHOWN} more cuts`)
  out.push('')
  return out
}

const splitSteps = kinds => {
  const steps = [
    `Start from the proposed cuts, then move blocks between parts so each part has one responsibility: each new file has one reason to change and stays under ${TARGET} tokens. A single block over ${TARGET} tokens must be split too: extract its inner helpers, lookup tables and constants into their own modules first.`,
    'Keep the original path as the module entry: it exports the interface callers use, and nothing else. Move the implementation into a private folder named after the file (<dir>/<original>/<part>.<ext>, kebab-case). Move code verbatim; do not rewrite it during the split. Keep each function with the private helpers and types that only it uses. Put helpers that two parts use in a third file that both import. Do not create import cycles.',
    'Importers keep importing the entry; never import a file in the private folder from outside it. Exports that no importer outside the module uses are candidates to hide: run `rg -l "<original basename without extension>"` to find importers.',
  ]
  if (kinds.has('test'))
    steps.push(
      'Tests: test through the module entry, the same interface callers use. Split the test file by behaviour of the interface, one <module>-<behaviour>.test.<ext> per group. Move each test with the fixtures it uses; put fixtures shared by several test files in a <name>-fixtures.<ext> helper. Keep every assertion: the split must not drop coverage.',
    )
  if (kinds.has('style'))
    steps.push(
      'Styles: move each component or page region into an SCSS partial and `@use` it from the original file in the same position. Keep the rule order, because the cascade depends on it.',
    )
  if (kinds.has('docs'))
    steps.push(
      'Docs: split at ## headings into separate notes. Keep the original as an index that links to each part, so existing links and paths stay valid.',
    )
  steps.push(
    'Run the checks that cover the moved code (`pnpm exec tsgo --noEmit`, `pnpm exec oxlint <files>`, `pnpm test <affected tests>`), then `git add` the new files and the shortened original, and commit again.',
  )
  return [
    'How to split:',
    ...steps.map((s, i) => `  ${i + 1}. ${s}`),
    '',
    'Lockfiles, generated, vendored and human-authored paths are exempt through .gitattributes (linguist-generated, linguist-vendored, -agent-authored). Do not mark code that agents write as exempt; split it.',
    '',
  ]
}

const main = argv => {
  const json = argv.includes('--json')
  const all = argv.includes('--all')
  const at = argv.indexOf('--paths')
  const paths = at === -1 ? undefined : argv.slice(at + 1).filter(a => !a.startsWith('--'))
  const source = all ? 'all' : paths ? 'paths' : 'staged'
  const entries = all ? indexEntries() : paths ? indexEntries(paths) : stagedEntries()
  const files = overBudget(entries)
    .map(describe)
    .sort((a, b) => b.tokens - a.tokens)

  if (json) {
    const report = { limit: LIMIT, target: TARGET, bytesPerToken: BYTES_PER_TOKEN, source, files }
    process.stdout.write(JSON.stringify(report, null, 2) + '\n')
    return 0
  }
  if (files.length === 0) return 0

  const plural = files.length === 1 ? 'file exceeds' : 'files exceed'
  const where = source === 'staged' ? 'staged' : 'tracked'
  const out = [
    '',
    `token budget: ${files.length} ${where} ${plural} ${LIMIT} tokens (tokens = bytes / ${BYTES_PER_TOKEN}, so ${MAX_BYTES} bytes).`,
    '',
  ]
  if (source === 'all') {
    for (const d of files)
      out.push(
        `${String(d.tokens).padStart(8)} tok  ${String(d.parts).padStart(3)} parts  ${d.path}`,
      )
    out.push('', 'Run with --paths <file> for a cut plan, or --json for the refactor harness.', '')
  } else {
    for (const d of files) out.push(...cutPlan(d))
    out.push(...splitSteps(new Set(files.map(d => d.kind))))
  }
  process.stderr.write(out.join('\n'))
  return 1
}

if (import.meta.main) process.exit(main(process.argv.slice(2)))
