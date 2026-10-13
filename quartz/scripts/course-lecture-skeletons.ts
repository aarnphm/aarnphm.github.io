// Writes an empty `lectures/<nn>.md` for every unit of a course that has none, so each unit
// opens the lecture workspace. The page is the `start notes` draft with no marks.
// Usage: pnpm exec tsx quartz/scripts/course-lecture-skeletons.ts [content/courses/<dir> ...]
import yaml from 'js-yaml'
import { existsSync } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { promoteDrafts } from '../components/scripts/course-promote'

interface UnitResource {
  id: string
  file?: string
  youtube?: string
}

interface Unit {
  n: number
  title: string
  resources?: UnitResource[]
}

const root = 'content/courses'
const pad = (n: number) => String(n).padStart(2, '0')
const today = new Date().toISOString().slice(0, 10)

async function courseDirs(args: string[]): Promise<string[]> {
  if (args.length > 0) return args.map(arg => path.basename(arg))
  const entries = await fs.readdir(root, { withFileTypes: true })
  return entries.filter(entry => entry.isDirectory()).map(entry => entry.name)
}

async function skeletons(dir: string): Promise<string[]> {
  const home = await fs.readFile(path.join(root, dir, 'index.md'), 'utf8')
  const head = home.match(/^---\n([\s\S]*?)\n---/)?.[1]
  const course = head
    ? (yaml.load(head) as { course?: { unitKind?: string; units?: Unit[] } }).course
    : undefined
  if (!course?.units?.length) return []
  const unitKind = course.unitKind ?? 'lecture'
  const written: string[] = []
  for (const unit of course.units) {
    const file = path.join(root, dir, 'lectures', `${pad(unit.n)}.md`)
    if (existsSync(file)) continue
    const resources = unit.resources ?? []
    // the lecture pdf when OCW ships one, else the first video
    const source =
      resources.find(r => r.file?.toLowerCase().endsWith('.pdf') && !/transcript/i.test(r.id)) ??
      resources.find(r => r.youtube)
    const resource =
      source && existsSync(path.join(root, dir, 'resources', source.id, 'index.md'))
        ? `courses/${dir}/resources/${source.id}`
        : undefined
    const { md } = await promoteDrafts(
      {
        dir,
        n: unit.n,
        unitKind,
        title: unit.title,
        resource,
        pdf: source?.file ? `courses/${dir}/${source.file}` : '',
        unit: `courses/${dir}/lectures/${pad(unit.n)}`,
        date: today,
      },
      [],
      [],
    )
    await fs.mkdir(path.dirname(file), { recursive: true })
    await fs.writeFile(file, md, { flag: 'wx' })
    written.push(file)
  }
  return written
}

for (const dir of await courseDirs(process.argv.slice(2))) {
  const written = await skeletons(dir)
  console.log(`${dir}: ${written.length} new lecture pages`)
}
