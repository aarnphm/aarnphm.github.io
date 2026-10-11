// WebVTT parsing and paragraph grouping, shared by the vendor script (markdown) and the watch
// page script (per-cue follow). No markdown dependencies here: this file ships to the browser.

/** One caption cue, seconds from the start of the video. */
export interface VttCue {
  start: number
  end: number
  text: string
}

export interface VttParagraph {
  start: number
  text: string
  /** The cues the paragraph joins, in order. */
  cues: VttCue[]
}

const timingRe =
  /^(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})\s+-->\s+(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})(?:\s|$)/
const entities: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&nbsp;': ' ',
  '&quot;': '"',
  '&#39;': "'",
  '&apos;': "'",
  '&lrm;': '',
  '&rlm;': '',
}
const speakerRe = /^[A-Z][A-Z .'-]{1,40}: /
const sentenceEndRe = /[.?!]["')\]]?$/
const GAP_SECONDS = 2
const SENTENCE_SECONDS = 30
const CAP_SECONDS = 60

function seconds(h: string | undefined, m: string, s: string, ms: string): number {
  return Number(h ?? 0) * 3600 + Number(m) * 60 + Number(s) + Number(ms) / 1000
}

function cueText(lines: string[]): string {
  const raw = lines.join(' ').replace(/<[^>]*>/g, '')
  return raw
    .replace(/&[a-z#0-9]+;/gi, entity => entities[entity.toLowerCase()] ?? entity)
    .replace(/\s+/g, ' ')
    .trim()
}

/** `[SQUEAKING]`, `[RUSTLING] [CLICKING]`: sound tags with no speech. */
function soundOnly(text: string): boolean {
  return /^(?:\[[^\]]*\]\s*)+$/.test(text)
}

/** Reads a WebVTT file. Blocks without a valid timing line are skipped and counted. */
export function parseVtt(text: string): { cues: VttCue[]; skipped: number } {
  const body = text.replace(/^﻿/, '').replace(/\r\n?/g, '\n')
  if (!/^WEBVTT(?:[ \t\n]|$)/.test(body)) return { cues: [], skipped: 0 }
  const cues: VttCue[] = []
  let skipped = 0
  for (const block of body.split(/\n{2,}/).slice(1)) {
    const lines = block.split('\n').filter(line => line.length > 0)
    if (lines.length === 0) continue
    if (/^(NOTE|STYLE|REGION)(?:\s|$)/.test(lines[0])) continue
    const index = lines.findIndex(line => line.includes('-->'))
    if (index === -1 || index > 1) continue
    const timing = timingRe.exec(lines[index])
    if (!timing) {
      skipped++
      continue
    }
    const start = seconds(timing[1], timing[2], timing[3], timing[4])
    const end = seconds(timing[5], timing[6], timing[7], timing[8])
    if (end < start) {
      skipped++
      continue
    }
    const content = cueText(lines.slice(index + 1))
    if (content.length === 0 || soundOnly(content)) continue
    cues.push({ start, end, text: content })
  }
  cues.sort((a, b) => a.start - b.start)
  return { cues, skipped }
}

/** Joins cues into paragraphs: a speaker change, a silence over 2 s, a sentence end after 30 s, or 60 s. */
export function groupCues(cues: VttCue[]): VttParagraph[] {
  const paragraphs: VttParagraph[] = []
  let current: { start: number; parts: string[]; cues: VttCue[] } | null = null
  let previous: VttCue | null = null
  for (const cue of cues) {
    const span = current ? cue.start - current.start : 0
    const breaks =
      !current ||
      !previous ||
      speakerRe.test(cue.text) ||
      cue.start - previous.end > GAP_SECONDS ||
      (span >= SENTENCE_SECONDS && sentenceEndRe.test(previous.text)) ||
      span >= CAP_SECONDS
    if (breaks) {
      if (current) paragraphs.push(close(current))
      current = { start: cue.start, parts: [cue.text], cues: [cue] }
    } else {
      current!.parts.push(cue.text)
      current!.cues.push(cue)
    }
    previous = cue
  }
  if (current) paragraphs.push(close(current))
  return paragraphs
}

function close(group: { start: number; parts: string[]; cues: VttCue[] }): VttParagraph {
  return { start: group.start, text: group.parts.join(' '), cues: group.cues }
}
