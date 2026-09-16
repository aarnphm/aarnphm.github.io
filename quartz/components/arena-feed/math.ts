const latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
const greek = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡϴΣΤΥΦΧΨΩ∇αβγδεζηθικλμνξοπρςστυφχψω∂ϵϑϰϕϱϖ'
const variants: Record<string, { latin: number; greek?: number; digits?: number }> = {
  bold: { latin: 0x1d400, greek: 0x1d6a8, digits: 0x1d7ce },
  italic: { latin: 0x1d434, greek: 0x1d6e2 },
  'bold-italic': { latin: 0x1d468, greek: 0x1d71c },
  script: { latin: 0x1d49c },
  'bold-script': { latin: 0x1d4d0 },
  fraktur: { latin: 0x1d504 },
  'double-struck': { latin: 0x1d538, digits: 0x1d7d8 },
  'bold-fraktur': { latin: 0x1d56c },
  'sans-serif': { latin: 0x1d5a0, digits: 0x1d7e2 },
  'bold-sans-serif': { latin: 0x1d5d4, greek: 0x1d756, digits: 0x1d7ec },
  'sans-serif-italic': { latin: 0x1d608 },
  'sans-serif-bold-italic': { latin: 0x1d63c, greek: 0x1d790 },
  monospace: { latin: 0x1d670, digits: 0x1d7f6 },
}

// Unicode leaves holes for letters already encoded in the Letterlike Symbols block.
const letterlike: Record<number, string> = {
  0x1d455: 'ℎ',
  0x1d49d: 'ℬ',
  0x1d4a0: 'ℰ',
  0x1d4a1: 'ℱ',
  0x1d4a3: 'ℋ',
  0x1d4a4: 'ℐ',
  0x1d4a7: 'ℒ',
  0x1d4a8: 'ℳ',
  0x1d4ad: 'ℛ',
  0x1d4ba: 'ℯ',
  0x1d4bc: 'ℊ',
  0x1d4c4: 'ℴ',
  0x1d506: 'ℭ',
  0x1d50b: 'ℌ',
  0x1d50c: 'ℑ',
  0x1d515: 'ℜ',
  0x1d51d: 'ℨ',
  0x1d53a: 'ℂ',
  0x1d53f: 'ℍ',
  0x1d545: 'ℕ',
  0x1d547: 'ℙ',
  0x1d548: 'ℚ',
  0x1d549: 'ℝ',
  0x1d551: 'ℤ',
}

export function mathVariantText(text: string, variant: string): string {
  const ranges = Object.hasOwn(variants, variant) ? variants[variant] : undefined
  if (!ranges) return text
  return Array.from(text, character => {
    const latinIndex = latin.indexOf(character)
    const greekIndex = greek.indexOf(character)
    const digitIndex = '0123456789'.indexOf(character)
    let point: number | undefined
    if (latinIndex >= 0) point = ranges.latin + latinIndex
    else if (ranges.greek !== undefined && greekIndex >= 0) point = ranges.greek + greekIndex
    else if (ranges.digits !== undefined && digitIndex >= 0) point = ranges.digits + digitIndex
    return point === undefined ? character : (letterlike[point] ?? String.fromCodePoint(point))
  }).join('')
}
