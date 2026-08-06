import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { gunzipSync } from 'node:zlib'
import type { QuartzConfig } from '../../cfg'
import type { BuildCtx } from '../../util/ctx'
import type { StaticResources } from '../../util/resources'
import { type FilePath, isFilePath } from '../../util/path'
import {
  AgentSkills,
  createAgentSkillsPublication,
  loadAgentSkillSources,
  type AgentSkillFile,
  type AgentSkillSource,
} from './agent-skills'

const testTheme = {
  typography: { header: 'system-ui', body: 'system-ui', code: 'monospace' },
  cdnCaching: false,
  colors: {
    lightMode: {
      light: '#ffffff',
      lightgray: '#eeeeee',
      gray: '#999999',
      darkgray: '#555555',
      dark: '#000000',
      secondary: '#000000',
      tertiary: '#000000',
      highlight: '#eeeeee',
      textHighlight: '#eeeeee',
    },
    darkMode: {
      light: '#000000',
      lightgray: '#222222',
      gray: '#999999',
      darkgray: '#dddddd',
      dark: '#ffffff',
      secondary: '#ffffff',
      tertiary: '#ffffff',
      highlight: '#222222',
      textHighlight: '#222222',
    },
  },
  fontOrigin: 'local',
} satisfies QuartzConfig['configuration']['theme']

const resources: StaticResources = { css: [], js: [], additionalHead: [] }

function filePath(value: string): FilePath {
  if (isFilePath(value)) return value
  throw new Error(`invalid file path ${value}`)
}

function testCtx(root: string): BuildCtx {
  return {
    buildId: 'test',
    argv: {
      directory: path.join(root, 'content'),
      verbose: false,
      output: path.join(root, 'public'),
      serve: false,
      watch: true,
      port: 8080,
      wsPort: 3001,
      force: false,
    },
    cfg: {
      configuration: {
        pageTitle: 'test',
        enableSPA: true,
        enablePopovers: true,
        analytics: null,
        ignorePatterns: [],
        defaultDateType: 'created',
        theme: testTheme,
        locale: 'en-US',
      },
      plugins: { transformers: [], filters: [], emitters: [] },
    },
    allSlugs: [],
    allFiles: [],
    incremental: true,
  }
}

async function collectEmitted(
  emitted: Promise<FilePath[]> | AsyncGenerator<FilePath> | null,
): Promise<FilePath[]> {
  const result = await emitted
  if (result === null) return []
  if (!(Symbol.asyncIterator in result)) return result

  const files: FilePath[] = []
  for await (const emittedFile of result) files.push(emittedFile)
  return files
}

function file(path: string, content: string): AgentSkillFile {
  return { path, content: Buffer.from(content), mode: 0o644 }
}

function source(name: string, description: string, supportingFiles: AgentSkillFile[] = []) {
  const skill = file(
    'SKILL.md',
    `---\nname: ${name}\ndescription: ${description}\n---\n\n# ${name}\n`,
  )
  return { directoryName: name, files: [skill, ...supportingFiles] }
}

function tarFiles(archive: Buffer): Map<string, Buffer> {
  const tar = gunzipSync(archive)
  const files = new Map<string, Buffer>()
  let offset = 0

  while (offset + 512 <= tar.byteLength) {
    const header = tar.subarray(offset, offset + 512)
    if (header.every(byte => byte === 0)) break
    const name = header.subarray(0, 100).toString().split('\0', 1)[0]
    const sizeField = header.subarray(124, 136).toString().split('\0', 1)[0].trim()
    const size = Number.parseInt(sizeField, 8)
    const start = offset + 512
    files.set(name, tar.subarray(start, start + size))
    offset = start + Math.ceil(size / 512) * 512
  }

  return files
}

test('publishes single-file skills with raw-byte SHA-256 digests', () => {
  const skill = source('code-review', 'Review code for defects.')
  const publication = createAgentSkillsPublication([skill])
  const artifact = publication.artifacts[0]
  const expectedDigest = createHash('sha256').update(skill.files[0].content).digest('hex')

  assert.deepEqual(publication.index, {
    $schema: 'https://schemas.agentskills.io/discovery/0.2.0/schema.json',
    skills: [
      {
        name: 'code-review',
        type: 'skill-md',
        description: 'Review code for defects.',
        url: '/.well-known/agent-skills/code-review/SKILL.md',
        digest: `sha256:${expectedDigest}`,
      },
    ],
  })
  assert.equal(artifact.ext, '.md')
  assert.equal(artifact.content, skill.files[0].content)
})

test('publishes skills with supporting files as deterministic root-level archives', () => {
  const skill = source('compiler-course', 'Answer compiler construction questions.', [
    file('references/parsing.md', '# Parsing\n'),
  ])
  const first = createAgentSkillsPublication([skill])
  const second = createAgentSkillsPublication([skill])
  const artifact = first.artifacts[0]
  const files = tarFiles(artifact.content)

  assert.equal(artifact.ext, '.tar.gz')
  assert.equal(artifact.content.equals(second.artifacts[0].content), true)
  assert.deepEqual([...files.keys()], ['SKILL.md', 'references/parsing.md'])
  assert.equal(files.get('references/parsing.md')?.toString(), '# Parsing\n')
  assert.equal(first.index.skills[0].type, 'archive')
  assert.equal(first.index.skills[0].url, '/.well-known/agent-skills/compiler-course.tar.gz')
  assert.equal(
    first.index.skills[0].digest,
    `sha256:${createHash('sha256').update(artifact.content).digest('hex')}`,
  )
})

test('rejects invalid names and missing skill metadata', () => {
  assert.throws(
    () => createAgentSkillsPublication([source('Invalid_Name', 'Invalid name.')]),
    /invalid agent skill name/,
  )

  const missingMetadata: AgentSkillSource = {
    directoryName: 'missing-metadata',
    files: [file('SKILL.md', '# Missing metadata\n')],
  }
  assert.throws(
    () => createAgentSkillsPublication([missingMetadata]),
    /requires name and description frontmatter/,
  )
})

test('partial emitter replaces the publication after skill source changes', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'quartz-agent-skills-partial-'))
  try {
    const sourceDirectory = path.join(root, 'skills')
    const skillDirectory = path.join(sourceDirectory, 'code-review')
    await mkdir(skillDirectory, { recursive: true })
    await writeFile(
      path.join(skillDirectory, 'SKILL.md'),
      '---\nname: code-review\ndescription: Review code.\n---\n',
    )

    const ctx = testCtx(root)
    const plugin = AgentSkills({ directory: sourceDirectory })
    const partialEmit = plugin.partialEmit
    assert.ok(partialEmit)
    await collectEmitted(plugin.emit(ctx, [], resources))

    const unrelated = await collectEmitted(
      partialEmit(ctx, [], resources, [
        { type: 'change', path: filePath(path.join(root, 'content', 'note.md')) },
      ]),
    )
    assert.deepEqual(unrelated, [])

    const staleOutput = path.join(ctx.argv.output, '.well-known/agent-skills/removed/SKILL.md')
    await mkdir(path.dirname(staleOutput), { recursive: true })
    await writeFile(staleOutput, 'stale')

    const emitted = await collectEmitted(
      partialEmit(ctx, [], resources, [
        { type: 'change', path: filePath(path.join(skillDirectory, 'SKILL.md')) },
      ]),
    )

    assert.equal(emitted.length, 2)
    await stat(path.join(ctx.argv.output, '.well-known/agent-skills/index.json'))
    await assert.rejects(stat(staleOutput))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('loads every tracked garden skill into an RFC-valid publication', async () => {
  const publication = createAgentSkillsPublication(await loadAgentSkillSources('.claude/skills'))
  assert.deepEqual(
    publication.index.skills.map(skill => skill.name),
    [
      'add-descriptions',
      'core',
      'flashcards',
      'interactive-diagrams',
      'quartz-plugins',
      'sfwr-4tb3',
    ],
  )
  assert.equal(
    publication.index.skills.every(skill => /^sha256:[0-9a-f]{64}$/.test(skill.digest)),
    true,
  )
})
