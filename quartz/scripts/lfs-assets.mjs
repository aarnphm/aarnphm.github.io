import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

const directory = path.dirname(fileURLToPath(import.meta.url))
const command = process.argv[2]
const usage =
  'usage: node quartz/scripts/lfs-assets.mjs audit|stage-images|upload-images|hydrate|verify-hydration [--repo PATH] [--report PATH]'
const commands = new Set(['audit', 'stage-images', 'upload-images', 'hydrate', 'verify-hydration'])
const imageExtensions = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.gif',
  '.avif',
  '.heic',
  '.svg',
])
const roots = ['content/thoughts/images', 'content/hinterland/app']
const required =
  'quartz/runtime/native/packs/**,content/triathlon/wind/**/*.json,content/thoughts/images/**,content/hinterland/app/**'

function run(repo, executable, args, options = {}) {
  return execFileSync(executable, args, {
    cwd: repo,
    encoding: 'utf8',
    timeout: 280_000,
    maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
    ...options,
    stdio: ['pipe', 'pipe', 'pipe'],
  })
}

function git(repo, args, options) {
  return run(repo, 'git', args, options)
}

function validate(repo) {
  return JSON.parse(
    run(repo, process.execPath, [path.join(directory, 'validate-deployment-assets.mjs'), ...roots]),
  )
}

function imagePaths(repo) {
  return [
    ...new Set(
      git(repo, ['ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', roots[0]])
        .split('\0')
        .filter(filename => imageExtensions.has(path.extname(filename).toLowerCase())),
    ),
  ].sort()
}

function pointers(repo) {
  const paths = imagePaths(repo)
  if (paths.length === 0) throw new Error('No image files found')
  const data = git(repo, ['cat-file', '--batch'], {
    input: paths.map(filename => `:${filename}\n`).join(''),
    encoding: null,
  })
  let offset = 0
  return paths.map(filename => {
    const end = data.indexOf(10, offset)
    const header = data.subarray(offset, end).toString().split(' ')
    if (header[1] !== 'blob')
      throw new Error(`${filename}: stage the image with stage-images first`)
    const size = Number(header[2])
    const content = data.subarray(end + 1, end + 1 + size).toString()
    offset = end + size + 2
    const match = content.match(
      /^version https:\/\/git-lfs.github.com\/spec\/v1\noid sha256:([a-f0-9]{64})\nsize (\d+)\n$/,
    )
    if (!match)
      throw new Error(`${filename}: indexed image is not an LFS pointer; run stage-images`)
    const bytes = readFileSync(path.join(repo, filename))
    const oid = createHash('sha256').update(bytes).digest('hex')
    if (oid !== match[1] || bytes.length !== Number(match[2]))
      throw new Error(`${filename}: working bytes differ from indexed LFS hash`)
    return { path: filename, oid, bytes: bytes.length, pointerBytes: size }
  })
}

function audit(repo) {
  const validation = validate(repo)
  const records = pointers(repo)
  return {
    status: 'passed',
    images: records.length,
    uniqueObjects: new Set(records.map(record => record.oid)).size,
    imageBytes: records.reduce((sum, record) => sum + record.bytes, 0),
    pointerBytes: records.reduce((sum, record) => sum + record.pointerBytes, 0),
    validation,
  }
}

function verifyHydration(repo) {
  if (git(repo, ['status', '--porcelain', '--', '.gitattributes', ...roots]).trim()) {
    throw new Error(
      'Commit image, app, and LFS attribute changes before verifying hydration from HEAD',
    )
  }
  const records = pointers(repo)
  const temporary = mkdtempSync(path.join(os.tmpdir(), 'garden-lfs-hydration-'))
  const clone = path.join(temporary, 'checkout')
  try {
    const environment = { ...process.env, GIT_TERMINAL_PROMPT: '0', GIT_LFS_SKIP_SMUDGE: '1' }
    let endpoint
    try {
      endpoint = git(repo, ['config', '--get', 'lfs.url']).trim()
    } catch (error) {
      if (error.status !== 1) throw error
      endpoint = git(repo, ['config', '--file', '.lfsconfig', '--get', 'lfs.url']).trim()
    }
    const index = Number(environment.GIT_CONFIG_COUNT ?? 0)
    const env = {
      ...environment,
      GIT_CONFIG_COUNT: String(index + 1),
      [`GIT_CONFIG_KEY_${index}`]: 'lfs.url',
      [`GIT_CONFIG_VALUE_${index}`]: endpoint,
    }
    git(repo, ['clone', '--shared', '--no-checkout', repo, clone], { env })
    git(clone, ['checkout', 'HEAD', '--', '.gitattributes', '.lfsconfig', ...roots], { env })
    git(clone, ['remote', 'set-url', 'origin', git(repo, ['remote', 'get-url', 'origin']).trim()], {
      env,
    })
    // Local-clone hooks can seed the LFS cache; remove only this temporary checkout's copy.
    const cache = path.join(clone, '.git/lfs/objects')
    rmSync(cache, { recursive: true, force: true })
    for (const record of records) {
      if (
        !readFileSync(path.join(clone, record.path), 'utf8').startsWith(
          'version https://git-lfs.github.com/spec/v1\n',
        )
      ) {
        throw new Error(`${record.path}: expected a pointer before hydration`)
      }
    }
    git(
      clone,
      [
        'lfs',
        'pull',
        '--include=content/thoughts/images/**,content/hinterland/app/**',
        '--exclude=',
      ],
      { env },
    )
    const result = audit(clone)
    if (JSON.stringify(pointers(clone)) !== JSON.stringify(records))
      throw new Error('Hydrated image inventory differs from source HEAD')
    return { ...result, emptyInitialCache: true, head: git(repo, ['rev-parse', 'HEAD']).trim() }
  } finally {
    rmSync(temporary, { recursive: true, force: true })
  }
}

try {
  if (['help', '--help', '-h'].includes(command)) {
    process.stdout.write(usage + '\n')
    process.exit(0)
  }
  if (!commands.has(command)) throw new Error(usage)
  const { values } = parseArgs({
    args: process.argv.slice(3),
    options: { repo: { type: 'string' }, report: { type: 'string' } },
  })
  const repo = path.resolve(values.repo ?? path.join(directory, '../..'))
  let result
  if (command === 'stage-images') {
    validate(repo)
    const paths = ['.gitattributes', ...imagePaths(repo)]
    const attribution = 'content/thoughts/images/instagram/attribution.json'
    try {
      statSync(path.join(repo, attribution))
      paths.push(attribution)
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
    const input = paths.join('\0') + '\0'
    git(repo, ['add', '--pathspec-from-file=-', '--pathspec-file-nul'], { input })
    git(repo, ['add', '--renormalize', '--pathspec-from-file=-', '--pathspec-file-nul'], { input })
    result = audit(repo)
  } else if (command === 'upload-images') {
    result = audit(repo)
    const oids = [...new Set(pointers(repo).map(record => record.oid))]
    git(repo, ['lfs', 'push', '--object-id', 'origin', '--stdin'], {
      input: oids.join('\n') + '\n',
    })
    result.uploadedObjects = oids.length
  } else if (command === 'hydrate') {
    git(repo, ['lfs', 'pull', `--include=${required}`, '--exclude='])
    result = audit(repo)
  } else if (command === 'verify-hydration') result = verifyHydration(repo)
  else result = audit(repo)
  result = { command, repo, ...result }
  const json = JSON.stringify(result, null, 2) + '\n'
  if (values.report) writeFileSync(path.resolve(values.report), json)
  process.stdout.write(json)
} catch (error) {
  const detail = error instanceof Error ? error.message : String(error)
  process.stderr.write(
    detail.replace(/https:\/\/[^\s/@]+:[^\s/@]+@/g, 'https://[redacted]@') + '\n',
  )
  process.exitCode = 1
}
