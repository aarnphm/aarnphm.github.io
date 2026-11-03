import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import yaml from 'js-yaml'

const root = fileURLToPath(new URL('../../', import.meta.url))
const library = path.join(root, 'content/library')
const list = (value) => value == null ? [] : Array.isArray(value) ? value : [value]
const key = (value) => String(value).normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase().replace(/[^\p{L}\p{N}]/gu, '')
const read = (file) => {
  const text = fs.readFileSync(file, 'utf8')
  const match = text.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)
  if (!match) throw new Error(`Missing frontmatter: ${file}`)
  return yaml.load(match[1])
}
const entries = fs.readdirSync(library).filter((name) => name.endsWith('.md') && name !== 'index.md')
  .map((name) => ({ file: name, ...read(path.join(library, name)) }))
const categories = [...new Set(entries.flatMap((entry) => list(entry.category)))].sort()
const [command = 'context', argument] = process.argv.slice(2)
const output = (value) => console.log(JSON.stringify(value, null, 2))
const matches = (names) => entries.filter((entry) => [entry.title, entry.id, ...list(entry.aliases)].some((name) => names.some((candidate) => key(name) === key(candidate))))

function validate(book) {
  const errors = []
  for (const field of ['id', 'title', 'date', 'description', 'status', 'posters']) {
    if (typeof book[field] !== 'string' || !book[field].trim()) errors.push(`${field}: required nonempty string`)
  }
  for (const field of ['author', 'category', 'subcategory', 'tags']) {
    if (!Array.isArray(book[field]) || !book[field].length || book[field].some((item) => typeof item !== 'string' || !item.trim())) errors.push(`${field}: required nonempty string list`)
  }
  if (!Number.isInteger(book.year)) errors.push('year: verified publication year required')
  if (!/^\d{4}-\d{2}-\d{2}$/.test(book.date ?? '') || Number.isNaN(Date.parse(book.date))) errors.push('date: expected YYYY-MM-DD')
  if (book.status !== 'to read') errors.push('status: new intake defaults to "to read"; change existing reading state manually')
  if (list(book.tags).some((tag) => ['evergreen', 'meta'].includes(tag))) errors.push('tags: evergreen/meta exclude the note from antilibrary')
  if (list(book.category).some((category) => !categories.includes(category))) errors.push(`category: choose from ${categories.join(', ')}`)
  if (typeof book.id === 'string' && (/[\\/\x00-\x1f]/.test(book.id) || ['.', '..', 'index'].includes(book.id))) errors.push('id: invalid filename')
  if (book.finished != null) errors.push('finished: leave null for unread books')
  const poster = book.posters?.match(/^\[\[(library\/posters\/[^|\]]+)(?:\|[^\]]+)?\]\]$/)?.[1]
  if (!poster || !fs.existsSync(path.join(root, 'content', poster))) errors.push('posters: expected existing library/posters wikilink')
  if (typeof book.external !== 'string' || !/^https?:\/\//.test(book.external)) errors.push('external: verified publisher, author, or catalog URL required')
  const duplicates = matches([book.title, book.id, ...list(book.aliases)].filter(Boolean))
  if (duplicates.length) errors.push(`existing notes: ${duplicates.map((entry) => entry.file).join(', ')}`)
  return errors
}

if (command === 'context') {
  output({ count: entries.length, categories, statuses: [...new Set(entries.map((entry) => entry.status))], fields: ['id', 'title', 'author', 'year', 'date', 'description', 'category', 'subcategory', 'tags', 'status', 'posters', 'external', 'language', 'translator', 'notebook', 'aliases', 'finished', 'pdf'], rules: 'Read harness/library-intake/README.md. Run search before researching a new title.' })
} else if (command === 'search') {
  if (!argument) throw new Error('Usage: intake.mjs search "title or author"')
  output(entries.filter((entry) => [entry.title, entry.id, ...list(entry.aliases), ...list(entry.author)].some((value) => key(value).includes(key(argument)))).map(({ file, title, author, status, category, external }) => ({ file, title, author, status, category, external })))
} else if (command === 'render') {
  if (!argument) throw new Error('Usage: intake.mjs render /absolute/path/book.json')
  const book = JSON.parse(fs.readFileSync(argument, 'utf8'))
  const errors = validate(book)
  if (errors.length) { output({ errors }); process.exitCode = 1 } else {
    const { body = '', ...metadata } = book
    const complete = { language: null, translator: null, notebook: null, aliases: [], finished: null, pdf: null, ...metadata }
    console.log(`---\n${yaml.dump(complete, { lineWidth: -1, noRefs: true }).trimEnd()}\n---\n\n${body}`)
  }
} else {
  throw new Error('Commands: context | search QUERY | render BOOK.json')
}
