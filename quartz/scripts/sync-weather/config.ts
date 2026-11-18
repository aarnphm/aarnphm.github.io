import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import { promisify } from 'node:util'
import { type WeatherKitConfig } from '../../util/weather-kit'

const execFileAsync = promisify(execFile)

const KEYCHAIN_SERVICES = ['garden-weatherkit', 'WeatherKit', 'weatherkit']

async function readPrivateKey(): Promise<string | null> {
  const inline = process.env.WEATHERKIT_PRIVATE_KEY?.trim()
  if (inline) return inline.replaceAll('\\n', '\n')
  const file = process.env.WEATHERKIT_PRIVATE_KEY_FILE?.trim()
  if (!file) return null
  return fs.readFile(file, 'utf8')
}

async function keychainPassword(accounts: string[]): Promise<string | null> {
  for (const service of KEYCHAIN_SERVICES)
    for (const account of accounts) {
      try {
        const { stdout } = await execFileAsync(
          '/usr/bin/security',
          ['find-generic-password', '-w', '-s', service, '-a', account],
          { timeout: 5_000 },
        )
        const value = stdout.trim()
        if (value) return value
      } catch {}
    }
  return null
}

async function envOrKeychain(name: string, aliases: string[]): Promise<string | null> {
  const value = process.env[name]?.trim()
  if (value) return value
  return keychainPassword([name, ...aliases])
}

async function readMachinePrivateKey(): Promise<string | null> {
  const inline = await keychainPassword(['WEATHERKIT_PRIVATE_KEY', 'privateKey', 'private-key'])
  if (inline) return inline.replaceAll('\\n', '\n')
  const file = await keychainPassword([
    'WEATHERKIT_PRIVATE_KEY_FILE',
    'privateKeyFile',
    'private-key-file',
  ])
  if (!file) return null
  return fs.readFile(file, 'utf8')
}

export async function weatherKitConfig(): Promise<WeatherKitConfig | null> {
  const teamId = await envOrKeychain('WEATHERKIT_TEAM_ID', ['teamId', 'team-id'])
  const serviceId = await envOrKeychain('WEATHERKIT_SERVICE_ID', ['serviceId', 'service-id'])
  const keyId = await envOrKeychain('WEATHERKIT_KEY_ID', ['keyId', 'key-id'])
  const privateKey = (await readPrivateKey()) ?? (await readMachinePrivateKey())
  if (!teamId || !serviceId || !keyId || !privateKey) return null
  return { teamId, serviceId, keyId, privateKey }
}
