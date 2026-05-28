import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'
import type { FilePath } from './path'
import { joinSegments } from './path'

function runCpio(cwd: string, output: string, input: Buffer): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn('cpio', ['-0', '-pdl', output], { cwd, stdio: ['pipe', 'ignore', 'pipe'] })
    const stderr: Buffer[] = []
    child.stderr.on('data', chunk => stderr.push(chunk))
    child.on('error', reject)
    child.on('close', code => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(Buffer.concat(stderr).toString().trim() || `cpio exited with ${code}`))
      }
    })
    child.stdin.end(input)
  })
}

export async function batchHardlinkRelativeFiles(
  cwd: string,
  output: string,
  files: readonly FilePath[],
): Promise<FilePath[]> {
  if (files.length === 0) return []
  await fs.mkdir(output, { recursive: true })
  await runCpio(cwd, path.resolve(output), Buffer.from(`${files.join('\0')}\0`))
  return files.map(fp => joinSegments(output, fp) as FilePath)
}
