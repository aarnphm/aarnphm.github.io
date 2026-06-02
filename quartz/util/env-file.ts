import fs from 'node:fs/promises'

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export async function upsertEnvLine(file: string, key: string, value: string): Promise<void> {
  let content = ''
  try {
    content = await fs.readFile(file, 'utf8')
  } catch {
    content = ''
  }

  const line = `${key}=${value}`
  const pattern = new RegExp(`^${escapeRegex(key)}=.*$`, 'm')
  const next = pattern.test(content)
    ? content.replace(pattern, line)
    : `${content.trimEnd()}${content.trimEnd() ? '\n' : ''}${line}\n`
  await fs.writeFile(file, next)
}

export async function removeEnvKeys(file: string, keys: readonly string[]): Promise<void> {
  let content = ''
  try {
    content = await fs.readFile(file, 'utf8')
  } catch {
    return
  }

  const remove = new Set(keys)
  const next = content
    .split(/\r?\n/)
    .filter(line => {
      const eq = line.indexOf('=')
      return eq < 0 || !remove.has(line.slice(0, eq))
    })
    .join('\n')
    .trimEnd()
  await fs.writeFile(file, next ? `${next}\n` : '')
}
