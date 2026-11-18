#!/usr/bin/env -S tsx
export {}
if (process.argv.includes('--compatibility-fixtures')) {
  const { runFixturePreview } = await import('../compatibility/serve-fixtures')
  await runFixturePreview(process.argv.slice(2))
} else if (
  process.argv.includes('--help') ||
  process.argv.includes('-h') ||
  process.argv.includes('--kill')
) {
  const { resolveRuntimeConfig } = await import('./dev/config')
  const config = resolveRuntimeConfig(process.argv.slice(2))
  if (config.kill) {
    const { killDaemon } = await import('./dev/daemon')
    const { fileURLToPath } = await import('node:url')
    const { resolve } = await import('node:path')
    await killDaemon(
      process.env.GIT_ROOT
        ? resolve(process.env.GIT_ROOT)
        : fileURLToPath(new URL('../../', import.meta.url)),
    )
  }
} else {
  await import('./dev/manager')
}
