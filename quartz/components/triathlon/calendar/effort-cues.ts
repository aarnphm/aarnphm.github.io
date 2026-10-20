// Effort cues the coach writes in TrainingPeaks descriptions and notes: RPE ("7/10", "RPE 3"),
// intensity percentages ("@85-95%", "68-78% effort", "HR <70%"), zones and named efforts.
// The order resolves overlaps: the longest written form of a cue comes first.
const CUE = new RegExp(
  [
    String.raw`\bRPE\s*\d{1,2}(?:\s*[-–]\s*\d{1,2})?\b`,
    String.raw`(?<![\d.])\d{1,2}(?:\.\d)?(?:\s*[-–]\s*\d{1,2}(?:\.\d)?)?\s*\/\s*10\b(?:\s+effort\b)?`,
    String.raw`\bfat\s*max\b`,
    String.raw`\bsweet[\s-]?spot\b`,
    String.raw`\bHR\s*[<>≤≥]\s*\d{1,3}\s*%`,
    String.raw`(?<![\w.])\d{1,3}(?:\s*[-–]\s*\d{1,3})?\s*%\s*(?:of\s+)?(?:FTP|max|threshold|effort)\b`,
    String.raw`(?<=(?:@|\bat)\s*)\d{1,3}(?:\s*[-–]\s*\d{1,3})?\s*%`,
    String.raw`\bz(?:one)?\s?[1-7][abc]?\b`,
    String.raw`\b(?:talk|easy|steady|moderate|aerobic|comfortable|race|threshold|tempo|ironman|70\.3)\s+(?:pace|effort)\b`,
  ].join('|'),
  'gi',
)

export type EffortCueSegment = { text: string; cue: boolean }

/** Splits text into plain runs and effort cues, keeping every character in order. */
export const effortCueSegments = (text: string): EffortCueSegment[] => {
  const segments: EffortCueSegment[] = []
  let last = 0
  for (const match of text.matchAll(CUE)) {
    const start = match.index ?? 0
    if (start > last) segments.push({ text: text.slice(last, start), cue: false })
    segments.push({ text: match[0], cue: true })
    last = start + match[0].length
  }
  if (last < text.length) segments.push({ text: text.slice(last), cue: false })
  return segments
}

/** The distinct cues in reading order, with whitespace collapsed. */
export const effortCues = (...texts: readonly string[]): string[] => {
  const seen = new Map<string, string>()
  for (const text of texts)
    for (const segment of effortCueSegments(text)) {
      if (!segment.cue) continue
      const cue = segment.text.replace(/\s+/g, ' ')
      // "7/10 effort" and a later "7/10" name one cue.
      const key = cue.toLowerCase().replace(/\s+effort$/, '')
      if (!seen.has(key)) seen.set(key, cue)
    }
  return [...seen.values()]
}
