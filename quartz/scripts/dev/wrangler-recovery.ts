import { stripAnsi } from './log'

// Wrangler 4.151 can stop scheduling token renewals after an edge-preview API failure.
export function createWranglerPreviewFailureDetector(): (chunk: string) => boolean {
  let buffer = ''
  let detected = false
  return chunk => {
    if (detected) return false
    buffer = `${buffer}${chunk}`.slice(-8192)
    detected =
      /\[ERROR\] A request to the Cloudflare API \([^\n]*\/edge-preview\) failed\.[^[]*\[code: 10013\]/.test(
        stripAnsi(buffer),
      )
    return detected
  }
}
