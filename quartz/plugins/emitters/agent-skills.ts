import matter from 'gray-matter'
import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { gzipSync } from 'node:zlib'
import type { ChangeEvent, QuartzEmitterPlugin } from '../../types/plugin'
import type { BuildCtx } from '../../util/ctx'
import { isRecord } from '../../util/type-guards'
import { resetWriteCache, write } from './helpers'

export const AGENT_SKILLS_SOURCE_DIRECTORY = '.claude/skills'

const AGENT_SKILLS_SCHEMA = 'https://schemas.agentskills.io/discovery/0.2.0/schema.json'
const DISCOVERY_ROOT = '.well-known/agent-skills'
const SKILL_NAME_PATTERN = /^(?!-)(?!.*--)[a-z0-9-]{1,64}(?<!-)$/
const TAR_BLOCK_SIZE = 512

type AgentSkillType = 'skill-md' | 'archive'

export type AgentSkillIndexEntry = {
  name: string
  type: AgentSkillType
  description: string
  url: string
  digest: string
}

export type AgentSkillIndex = { $schema: string; skills: AgentSkillIndexEntry[] }

export type AgentSkillFile = { path: string; content: Buffer; mode: number }

export type AgentSkillSource = { directoryName: string; files: AgentSkillFile[] }

export type AgentSkillArtifact = { slug: string; ext: '.md' | '.tar.gz'; content: Buffer }

export type AgentSkillsPublication = { index: AgentSkillIndex; artifacts: AgentSkillArtifact[] }

type AgentSkillsOptions = { directory?: string }

function sha256(content: Buffer): string {
  return `sha256:${createHash('sha256').update(content).digest('hex')}`
}

function writeTarString(target: Buffer, offset: number, length: number, value: string): void {
  const content = Buffer.from(value)
  if (content.byteLength > length) {
    throw new Error(`tar field exceeds ${length} bytes: ${value}`)
  }
  content.copy(target, offset)
}

function writeTarOctal(target: Buffer, offset: number, length: number, value: number): void {
  const octal = value.toString(8)
  if (octal.length > length - 1) {
    throw new Error(`tar numeric field exceeds ${length - 1} digits: ${value}`)
  }
  writeTarString(target, offset, length, `${octal.padStart(length - 1, '0')}\0`)
}

function tarHeader(file: AgentSkillFile): Buffer {
  if (
    file.path.startsWith('/') ||
    file.path.split('/').includes('..') ||
    Buffer.byteLength(file.path) > 100
  ) {
    throw new Error(`unsupported skill archive path: ${file.path}`)
  }

  const header = Buffer.alloc(TAR_BLOCK_SIZE)
  writeTarString(header, 0, 100, file.path)
  writeTarOctal(header, 100, 8, file.mode & 0o777)
  writeTarOctal(header, 108, 8, 0)
  writeTarOctal(header, 116, 8, 0)
  writeTarOctal(header, 124, 12, file.content.byteLength)
  writeTarOctal(header, 136, 12, 0)
  header.fill(0x20, 148, 156)
  writeTarString(header, 156, 1, '0')
  writeTarString(header, 257, 6, 'ustar\0')
  writeTarString(header, 263, 2, '00')
  writeTarString(header, 265, 32, 'aarnphm')
  writeTarString(header, 297, 32, 'aarnphm')

  const checksum = header.reduce((sum, byte) => sum + byte, 0)
  const checksumOctal = checksum.toString(8)
  if (checksumOctal.length > 6) {
    throw new Error(`tar checksum exceeds 6 digits: ${checksum}`)
  }
  writeTarString(header, 148, 8, `${checksumOctal.padStart(6, '0')}\0 `)
  return header
}

export function createAgentSkillArchive(files: readonly AgentSkillFile[]): Buffer {
  const blocks: Buffer[] = []
  for (const file of files) {
    blocks.push(tarHeader(file), file.content)
    const padding = (TAR_BLOCK_SIZE - (file.content.byteLength % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE
    if (padding > 0) blocks.push(Buffer.alloc(padding))
  }
  blocks.push(Buffer.alloc(TAR_BLOCK_SIZE * 2))
  return gzipSync(Buffer.concat(blocks), { level: 9 })
}

function skillMetadata(source: AgentSkillSource): { name: string; description: string } {
  const skillFile = source.files.find(file => file.path === 'SKILL.md')
  if (!skillFile) {
    throw new Error(`agent skill ${source.directoryName} is missing SKILL.md`)
  }

  const data: unknown = matter(skillFile.content).data
  if (!isRecord(data) || typeof data.name !== 'string' || typeof data.description !== 'string') {
    throw new Error(`agent skill ${source.directoryName} requires name and description frontmatter`)
  }

  const name = data.name.trim()
  const description = data.description.trim()
  if (!SKILL_NAME_PATTERN.test(name)) {
    throw new Error(`invalid agent skill name: ${name}`)
  }
  if (name !== source.directoryName) {
    throw new Error(`agent skill directory ${source.directoryName} does not match name ${name}`)
  }
  if (description.length === 0 || description.length > 1024) {
    throw new Error(`agent skill ${name} description must contain 1 to 1024 characters`)
  }
  return { name, description }
}

export function createAgentSkillsPublication(
  sources: readonly AgentSkillSource[],
): AgentSkillsPublication {
  const entries: AgentSkillIndexEntry[] = []
  const artifacts: AgentSkillArtifact[] = []
  const names = new Set<string>()

  for (const source of [...sources].sort((left, right) =>
    left.directoryName.localeCompare(right.directoryName),
  )) {
    const { name, description } = skillMetadata(source)
    if (names.has(name)) throw new Error(`duplicate agent skill name: ${name}`)
    names.add(name)

    const files = [...source.files].sort((left, right) => {
      if (left.path === 'SKILL.md') return -1
      if (right.path === 'SKILL.md') return 1
      return left.path.localeCompare(right.path)
    })
    const skillFile = files.find(file => file.path === 'SKILL.md')
    if (!skillFile) throw new Error(`agent skill ${name} is missing SKILL.md`)

    const singleFile = files.length === 1
    const type: AgentSkillType = singleFile ? 'skill-md' : 'archive'
    const ext = singleFile ? '.md' : '.tar.gz'
    const slug = singleFile ? `${DISCOVERY_ROOT}/${name}/SKILL` : `${DISCOVERY_ROOT}/${name}`
    const content = singleFile ? skillFile.content : createAgentSkillArchive(files)
    const url = `/${slug}${ext}`

    artifacts.push({ slug, ext, content })
    entries.push({ name, type, description, url, digest: sha256(content) })
  }

  return { index: { $schema: AGENT_SKILLS_SCHEMA, skills: entries }, artifacts }
}

async function readSkillFiles(
  directory: string,
  relativeDirectory = '',
): Promise<AgentSkillFile[]> {
  const entries = await fs.readdir(path.join(directory, relativeDirectory), { withFileTypes: true })
  const files: AgentSkillFile[] = []

  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const relativePath = path.posix.join(relativeDirectory, entry.name)
    const absolutePath = path.join(directory, relativePath)
    if (entry.isSymbolicLink()) {
      throw new Error(`agent skill archives cannot contain symbolic links: ${relativePath}`)
    }
    if (entry.isDirectory()) {
      files.push(...(await readSkillFiles(directory, relativePath)))
      continue
    }
    if (!entry.isFile()) {
      throw new Error(`agent skill archives require regular files: ${relativePath}`)
    }
    const [content, stat] = await Promise.all([fs.readFile(absolutePath), fs.stat(absolutePath)])
    files.push({ path: relativePath, content, mode: stat.mode })
  }

  return files
}

export async function loadAgentSkillSources(directory: string): Promise<AgentSkillSource[]> {
  const entries = await fs.readdir(directory, { withFileTypes: true })
  const sources: AgentSkillSource[] = []

  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    if (entry.isSymbolicLink()) {
      throw new Error(`agent skill directories cannot be symbolic links: ${entry.name}`)
    }
    if (!entry.isDirectory()) continue
    sources.push({
      directoryName: entry.name,
      files: await readSkillFiles(path.join(directory, entry.name)),
    })
  }

  return sources
}

function hasAgentSkillChanges(changeEvents: readonly ChangeEvent[], directory: string): boolean {
  return changeEvents.some(changeEvent => {
    const changedPath = path.resolve(changeEvent.path)
    return changedPath === directory || changedPath.startsWith(`${directory}${path.sep}`)
  })
}

async function* writeAgentSkillsPublication(ctx: BuildCtx, publication: AgentSkillsPublication) {
  for (const artifact of publication.artifacts) {
    yield write({ ctx, ...artifact })
  }
  yield write({
    ctx,
    slug: `${DISCOVERY_ROOT}/index`,
    ext: '.json',
    content: `${JSON.stringify(publication.index, null, 2)}\n`,
  })
}

export const AgentSkills: QuartzEmitterPlugin<AgentSkillsOptions> = options => ({
  name: 'AgentSkills',
  async *partialEmit(ctx, _content, _resources, changeEvents) {
    const directory = path.resolve(options?.directory ?? AGENT_SKILLS_SOURCE_DIRECTORY)
    if (!hasAgentSkillChanges(changeEvents, directory)) return

    const publication = createAgentSkillsPublication(await loadAgentSkillSources(directory))
    await fs.rm(path.join(ctx.argv.output, DISCOVERY_ROOT), { recursive: true, force: true })
    resetWriteCache()
    yield* writeAgentSkillsPublication(ctx, publication)
  },
  async *emit(ctx) {
    const directory = path.resolve(options?.directory ?? AGENT_SKILLS_SOURCE_DIRECTORY)
    const publication = createAgentSkillsPublication(await loadAgentSkillSources(directory))
    yield* writeAgentSkillsPublication(ctx, publication)
  },
})
