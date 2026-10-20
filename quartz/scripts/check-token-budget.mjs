#!/usr/bin/env node
// Pre-commit token budget: rejects staged files over LIMIT tokens, where tokens = bytes / 4.
// It reads the staged blobs, so partial staging is measured as it will be committed.
// Skipped: gitlinks, symlinks, binary blobs, and paths that .gitattributes (index copy) marks
// linguist-generated, linguist-vendored or -agent-authored.
import { execFileSync } from 'node:child_process'
import path from 'node:path'

const LIMIT = 5000
// Size to aim for after a split, so the next few edits do not trip the hook again.
const TARGET = 4000
const BYTES_PER_TOKEN = 4
const MAX_BYTES = LIMIT * BYTES_PER_TOKEN
const CUTS_SHOWN = 40

const tokens = bytes => Math.ceil(bytes / BYTES_PER_TOKEN)
const git = (args, input) => execFileSync('git', args, { input, maxBuffer: 1 << 30 })

// `:<src mode> <dst mode> <src sha> <dst sha> <status>\0<path>\0` per entry.
const raw = git(['diff', '--cached', '--raw', '-z', '--no-renames', '--diff-filter=ACMRT'])
  .toString('utf8')
  .split('\0')
const staged = []
for (let i = 0; i + 1 < raw.length; i += 2) {
  const [, mode, , sha] = raw[i].split(' ')
  if (mode === '100644' || mode === '100755') staged.push({ path: raw[i + 1], sha })
}
if (staged.length === 0) process.exit(0)

const sizes = git(['cat-file', '--batch-check=%(objectsize)'], staged.map(f => f.sha).join('\n'))
  .toString()
  .trim()
  .split('\n')
const oversized = staged
  .map((f, i) => ({ ...f, bytes: Number(sizes[i]) }))
  .filter(f => f.bytes > MAX_BYTES)
if (oversized.length === 0) process.exit(0)

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

const isSet = v => v === 'set' || v === 'true'
const isUnset = v => v === 'unset' || v === 'false'
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
if (violations.length === 0) process.exit(0)

const kindOf = p => {
  const base = path.basename(p)
  if (/\.(test|spec)\.[cm]?[jt]sx?$|(^test_.*|_test)\.py$/.test(base)) return 'test'
  if (/\.(s?css|sass|less)$/.test(base)) return 'style'
  if (/\.(md|mdx|markdown)$/.test(base)) return 'docs'
  return 'code'
}

// Top-level blocks: declarations and rules for code and styles, headings for prose.
// Comments and decorators directly above a declaration belong to it.
const blocks = (text, kind) => {
  const lines = text.split('\n')
  const starts = []
  let fenced = false
  lines.forEach((line, i) => {
    if (kind === 'docs') {
      if (/^(```|~~~)/.test(line)) fenced = !fenced
      if (!fenced && /^#{1,3} /.test(line)) starts.push(i)
    } else if (
      /^[^\s})\]'"`@/*#]|^@(media|container|supports|layer|mixin|function|include|keyframes)\b/.test(
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
    const label = lines
      .slice(start, end + 1)
      .find(l => /^[^\s/*#@]|^#{1,3} |^@(media|container)/.test(l))
    return {
      start: start + 1,
      end: end + 1,
      tokens: tokens(bytes),
      label: (label ?? lines[start]).trim().slice(0, 64),
    }
  })
}

// Proposed cuts: consecutive blocks packed up to TARGET tokens, so each part keeps neighbouring
// (usually related) code together and styles keep their cascade order.
const cuts = blocks => {
  const parts = []
  for (const b of blocks) {
    const last = parts.at(-1)
    if (last && !last.over && last.tokens + b.tokens <= TARGET) {
      last.end = b.end
      last.tokens += b.tokens
      last.count++
    } else parts.push({ ...b, count: 1, over: b.tokens > TARGET })
  }
  return parts
}

const out = []
const plural = violations.length === 1 ? 'file exceeds' : 'files exceed'
out.push(
  '',
  `token budget: ${violations.length} staged ${plural} ${LIMIT} tokens (tokens = bytes / ${BYTES_PER_TOKEN}, so ${MAX_BYTES} bytes).`,
  '',
)
for (const f of violations) {
  const t = tokens(f.bytes)
  const parts = Math.ceil(t / TARGET)
  out.push(
    `${f.path}: ${t} tokens. Split it into at least ${parts} files of at most ${TARGET} tokens each.`,
  )
  const plan = cuts(blocks(f.text, kindOf(f.path)))
  out.push('  proposed cuts (line ranges of consecutive top-level blocks):')
  for (const [i, c] of plan.slice(0, CUTS_SHOWN).entries()) {
    const range = `L${c.start}-${c.end}`.padEnd(14)
    const note = c.over ? '  <- one block over budget: split inside it' : ''
    out.push(
      `    ${String(i + 1).padStart(3)}. ${range}${String(c.tokens).padStart(6)} tok  ${c.label}${note}`,
    )
  }
  if (plan.length > CUTS_SHOWN) out.push(`    ... ${plan.length - CUTS_SHOWN} more cuts`)
  out.push('')
}

const kinds = new Set(violations.map(f => kindOf(f.path)))
const steps = [
  `Start from the proposed cuts, then move blocks between parts so each part has one responsibility: each new file has one reason to change and stays under ${TARGET} tokens. A single block over ${TARGET} tokens must be split too: extract its inner helpers, lookup tables and constants into their own modules first.`,
  'Create the new files next to the original, named <original>-<part>.<ext> in kebab-case, or create a directory <original>/ with one file per part. Move code verbatim; do not rewrite it during the split. Keep each function with the private helpers and types that only it uses. Put helpers that two parts use in a third module that both import. Do not create import cycles.',
  'Update every importer: find them with `rg -l "<original basename without extension>"` and point each import at the module that now owns the symbol. Keep the original path as a thin re-export only when many importers or an external contract depend on it.',
]
if (kinds.has('test'))
  steps.push(
    'Tests: split the test file along the modules under test, one <module>.test.<ext> per new module. Move each test with the fixtures it uses; put fixtures shared by several test files in a <name>-fixtures.<ext> helper. Keep every assertion: the split must not drop coverage.',
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

out.push('How to split:')
steps.forEach((s, i) => out.push(`  ${i + 1}. ${s}`))
out.push(
  '',
  'Lockfiles, generated, vendored and human-authored paths are exempt through .gitattributes (linguist-generated, linguist-vendored, -agent-authored). Do not mark code that agents write as exempt; split it.',
  '',
)
process.stderr.write(out.join('\n'))
process.exit(1)
