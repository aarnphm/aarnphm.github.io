import { createHash } from 'node:crypto'

export const hashContent = (value: unknown): string =>
  createHash('sha256').update(JSON.stringify(value)).digest('base64url')
