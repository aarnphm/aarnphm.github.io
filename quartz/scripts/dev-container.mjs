import { spawn, spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

export const devContainerName = 'quartz-dev'
export const devNodeImage = 'node:24-bookworm'
export const devPnpmVersion = '11.1.3'
export const devWorkspaceVolume = 'quartz-dev-workspace'
export const devNodeModulesVolume = 'quartz-dev-node24-node-modules'
export const devPnpmStoreVolume = 'quartz-dev-node24-pnpm-store'
export const devPublicVolume = 'quartz-dev-public'
export const devQuartzCacheVolume = 'quartz-dev-quartz-cache'
export const devContainerVolumes = [
  devWorkspaceVolume,
  devNodeModulesVolume,
  devPnpmStoreVolume,
  devPublicVolume,
  devQuartzCacheVolume,
]
export const devSourceDirectories = ['content', 'quartz', 'worker', 'migrations', '@types']
export const devWorkspaceFiles = [
  '.npmrc',
  '.pnpmfile.cjs',
  'globals.d.ts',
  'index.d.ts',
  'package.json',
  'pnpm-lock.yaml',
  'pnpm-workspace.yaml',
  'quartz.config.ts',
  'quartz.layout.ts',
  'tsconfig.json',
  'worker-configuration.d.ts',
  'wrangler.toml',
]

export function createDevContainerRunArgs(workspacePath) {
  const envFile = `${workspacePath}/.env`
  const envFileArgs = existsSync(envFile) ? ['--env-file', envFile] : []
  const wranglerConfig = `${process.env.HOME ?? ''}/.config/.wrangler`
  const wranglerConfigArgs = existsSync(wranglerConfig)
    ? ['--mount', `type=bind,source=${wranglerConfig},target=/root/.config/.wrangler`]
    : []
  const sourceDirectoryArgs = devSourceDirectories.flatMap(directory =>
    bindMountArgs(`${workspacePath}/${directory}`, `/workspace/${directory}`, true),
  )
  const workspaceFileLinks = devWorkspaceFiles
    .filter(file => existsSync(`${workspacePath}/${file}`))
    .flatMap(file => [
      `rm -f ${quoteShell(file)}`,
      `ln -s ${quoteShell(`/host/${file}`)} ${quoteShell(file)}`,
    ])
  const startupCommand = [
    ...workspaceFileLinks,
    'corepack enable',
    `corepack prepare pnpm@${devPnpmVersion} --activate`,
    'pnpm install --frozen-lockfile --store-dir /pnpm/store',
    'pnpm exec tsx quartz/scripts/dev.ts --retry 999 --serve --watch',
  ].join(' && ')

  return [
    'run',
    '-d',
    '--name',
    devContainerName,
    '--init',
    '-m',
    '16G',
    '-c',
    '8',
    '-e',
    'CI=true',
    '-e',
    'COREPACK_ENABLE_DOWNLOAD_PROMPT=0',
    ...envFileArgs,
    '-p',
    '7373:7373',
    '-p',
    '7374:7374',
    '-p',
    '8080:8080',
    '--mount',
    `type=volume,source=${devWorkspaceVolume},target=/workspace`,
    '--mount',
    `type=bind,source=${workspacePath},target=/host,readonly`,
    ...sourceDirectoryArgs,
    '--mount',
    `type=volume,source=${devNodeModulesVolume},target=/workspace/node_modules`,
    '--mount',
    `type=volume,source=${devPnpmStoreVolume},target=/pnpm/store`,
    '--mount',
    `type=volume,source=${devPublicVolume},target=/workspace/public`,
    '--mount',
    `type=volume,source=${devQuartzCacheVolume},target=/workspace/quartz/.quartz-cache`,
    ...wranglerConfigArgs,
    '-w',
    '/workspace',
    devNodeImage,
    'sh',
    '-lc',
    startupCommand,
  ]
}

function bindMountArgs(source, target, readOnly) {
  if (!existsSync(source)) {
    return []
  }
  const suffix = readOnly ? ',readonly' : ''
  return ['--mount', `type=bind,source=${source},target=${target}${suffix}`]
}

function quoteShell(value) {
  return `'${value.replaceAll("'", "'\\''")}'`
}

export async function main() {
  for (const volume of devContainerVolumes) {
    ensureVolume(volume)
  }

  if (!removeStoppedContainer()) {
    return
  }

  await runContainer(createDevContainerRunArgs(process.cwd()))
}

function ensureVolume(name) {
  if (listValues(['volume', 'list', '--quiet']).includes(name)) {
    return
  }
  runChecked(['volume', 'create', name])
}

function removeStoppedContainer() {
  const runningContainers = listValues(['list', '--quiet'])
  if (runningContainers.includes(devContainerName)) {
    process.stdout.write(`${devContainerName} is already running\n`)
    process.stdout.write(`container logs -f ${devContainerName}\n`)
    return false
  }

  const containers = listValues(['list', '--all', '--quiet'])
  if (containers.includes(devContainerName)) {
    runChecked(['delete', devContainerName])
  }

  return true
}

function listValues(args) {
  const output = runOutput(args)
  return output
    .split(/\r?\n/)
    .map(value => value.trim())
    .filter(value => value.length > 0)
}

function runOutput(args) {
  const result = spawnSync('container', args, { encoding: 'utf8' })
  if (result.error) {
    throw result.error
  }
  if (result.status !== 0) {
    throw new Error(formatFailure(args, result.stderr))
  }
  return result.stdout
}

function runChecked(args) {
  const result = spawnSync('container', args, { encoding: 'utf8', stdio: 'pipe' })
  if (result.error) {
    throw result.error
  }
  if (result.status !== 0) {
    throw new Error(formatFailure(args, result.stderr))
  }
}

function runContainer(args) {
  return new Promise((resolve, reject) => {
    const child = spawn('container', args, { stdio: 'inherit' })
    child.once('error', reject)
    child.once('close', code => {
      if (code === 0) {
        resolve()
        return
      }
      reject(new Error(`container ${args.join(' ')} exited with code ${code ?? 'null'}`))
    })
  })
}

function formatFailure(args, stderr) {
  const detail = stderr.trim()
  if (detail.length === 0) {
    return `container ${args.join(' ')} failed`
  }
  return `container ${args.join(' ')} failed: ${detail}`
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    const message = error instanceof Error ? error.message : String(error)
    process.stderr.write(`${message}\n`)
    process.exitCode = 1
  })
}
