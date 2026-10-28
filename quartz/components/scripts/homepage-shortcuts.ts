export function homepageShortcutKeys(labels: readonly string[]): string[] {
  const occurrences = new Map<string, number>()
  const used = new Set<string>()
  return labels.map(label => {
    const initials = label
      .normalize('NFKD')
      .replace(/\p{M}/gu, '')
      .split(/\s+/u)
      .map(word => word.match(/[a-z]/iu)?.[0]?.toUpperCase() ?? '')
      .filter(Boolean)
    const first = initials[0]
    if (!first) return ''
    const occurrence = (occurrences.get(first) ?? 0) + 1
    occurrences.set(first, occurrence)
    let key = occurrence > 1 && initials.length > 1 ? initials.join('') : first.repeat(occurrence)
    let repeat = occurrence
    while (used.has(key)) key = first.repeat(repeat++)
    used.add(key)
    return key
  })
}
