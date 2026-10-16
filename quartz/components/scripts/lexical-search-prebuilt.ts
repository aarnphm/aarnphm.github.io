import type { LexicalDocumentIndex } from './lexical-search-index'
import type { LexicalSearchSpec } from './lexical-search-types'
import type { LexicalField } from './lexical-search-types'
import { encode } from '../../util/search-text'

export const LEXICAL_FIELDS: readonly LexicalField[] = [
  'title',
  'fileName',
  'content',
  'tags',
  'aliases',
]

export type BucketList = (number[] | null)[]
export type FieldMapEntry = [string, BucketList]

export type DecodedCompactLexicalIndex = {
  reg: number[]
  fields: Record<LexicalField, FieldMapEntry[]>
  uniqueBucketsCount: number
  totalKeysCount: number
}

const CHAR_TABLE: string[] = Array.from({ length: 256 }, (_, i) => String.fromCharCode(i))

class ByteWriter {
  private buf: Uint8Array
  private pos = 0

  constructor(initialCapacity = 8 * 1024 * 1024) {
    this.buf = new Uint8Array(initialCapacity)
  }

  private ensure(extra: number): void {
    if (this.pos + extra <= this.buf.length) return
    let nextLen = this.buf.length * 2
    while (nextLen < this.pos + extra) nextLen *= 2
    const next = new Uint8Array(nextLen)
    next.set(this.buf.subarray(0, this.pos))
    this.buf = next
  }

  writeVarint(value: number): void {
    this.ensure(8)
    let v = value
    while (v >= 0x80) {
      this.buf[this.pos++] = (v & 0x7f) | 0x80
      v = Math.floor(v / 128)
    }
    this.buf[this.pos++] = v & 0x7f
  }

  finish(): Uint8Array {
    return this.buf.slice(0, this.pos)
  }
}

export function extractOfficialExportArrays(index: LexicalDocumentIndex): {
  reg: number[]
  fields: Record<LexicalField, FieldMapEntry[]>
  rawExportParts: number
  rawExportBytes: number
} {
  const reg: number[] = []
  const fields: Record<LexicalField, FieldMapEntry[]> = {
    title: [],
    fileName: [],
    content: [],
    tags: [],
    aliases: [],
  }
  let rawExportParts = 0
  let rawExportBytes = 0

  index.export((key: string, data: string) => {
    if (!data) return
    rawExportParts++
    rawExportBytes += Buffer.byteLength(data, 'utf8')
    const split = key.split('.')
    if (split[split.length - 1] === 'json') split.pop()
    const fieldName = split.length > 2 ? split[0] : ''
    const ref = split.length > 2 ? split[2] : split[1]

    if (!fieldName && ref === 'reg') {
      const parsed: number[] = JSON.parse(data)
      for (let i = 0; i < parsed.length; i++) {
        reg.push(parsed[i])
      }
    } else if (
      fieldName &&
      ref === 'map' &&
      (fieldName === 'title' ||
        fieldName === 'fileName' ||
        fieldName === 'content' ||
        fieldName === 'tags' ||
        fieldName === 'aliases')
    ) {
      const parsed: FieldMapEntry[] = JSON.parse(data)
      const target = fields[fieldName]
      for (let i = 0; i < parsed.length; i++) {
        target.push(parsed[i])
      }
    }
  })

  return { reg, fields, rawExportParts, rawExportBytes }
}

export function encodeCompactLexicalIndex(index: LexicalDocumentIndex): {
  bytes: Uint8Array
  stats: {
    regCount: number
    uniqueBucketsCount: number
    totalKeysCount: number
    rawExportParts: number
    rawExportBytes: number
  }
} {
  const { reg, fields, rawExportParts, rawExportBytes } = extractOfficialExportArrays(index)

  // Count global bucket frequencies for optimal varint ID assignment
  const bucketFreq = new Map<string, { buckets: BucketList; count: number }>()
  const fieldKeyToSig: Record<LexicalField, Array<[string, string]>> = {
    title: [],
    fileName: [],
    content: [],
    tags: [],
    aliases: [],
  }

  let totalKeysCount = 0
  for (const field of LEXICAL_FIELDS) {
    const entries = fields[field]
    totalKeysCount += entries.length
    const list: Array<[string, string]> = []
    list.length = entries.length
    for (let i = 0; i < entries.length; i++) {
      const [key, buckets] = entries[i]
      const sig = JSON.stringify(buckets)
      const existing = bucketFreq.get(sig)
      if (existing) {
        existing.count++
      } else {
        bucketFreq.set(sig, { buckets, count: 1 })
      }
      list[i] = [key, sig]
    }
    list.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    fieldKeyToSig[field] = list
  }

  const sortedUniqueBuckets = Array.from(bucketFreq.entries()).sort(
    (a, b) => b[1].count - a[1].count,
  )
  const sigToBucketId = new Map<string, number>()
  for (let i = 0; i < sortedUniqueBuckets.length; i++) {
    sigToBucketId.set(sortedUniqueBuckets[i][0], i)
  }

  const writer = new ByteWriter(8 * 1024 * 1024)

  // 1. Encode sorted reg IDs
  const sortedReg = reg.slice().sort((a, b) => a - b)
  writer.writeVarint(sortedReg.length)
  let prevReg = 0
  for (let i = 0; i < sortedReg.length; i++) {
    const id = sortedReg[i]
    writer.writeVarint(i === 0 ? id : id - prevReg - 1)
    prevReg = id
  }

  // 2. Encode globally deduplicated buckets table
  writer.writeVarint(sortedUniqueBuckets.length)
  for (let u = 0; u < sortedUniqueBuckets.length; u++) {
    const buckets = sortedUniqueBuckets[u][1].buckets
    let mask = 0
    for (let b = 0; b < buckets.length; b++) {
      const slot = buckets[b]
      if (slot && slot.length > 0) {
        mask |= 1 << b
      }
    }
    writer.writeVarint(mask)
    for (let b = 0; b < buckets.length; b++) {
      const slot = buckets[b]
      if (slot && slot.length > 0) {
        writer.writeVarint(slot.length)
        let prevId = slot[0]
        writer.writeVarint(prevId)
        for (let j = 1; j < slot.length; j++) {
          const curId = slot[j]
          if (curId <= prevId) {
            throw new Error(`Non-monotonic posting ID in bucket ${b}: ${curId} <= ${prevId}`)
          }
          writer.writeVarint(curId - prevId - 1)
          prevId = curId
        }
      }
    }
  }

  // 3. Encode per-field sorted keys with front-coded UTF-16 code units + bucket ID
  for (const field of LEXICAL_FIELDS) {
    const list = fieldKeyToSig[field]
    writer.writeVarint(list.length)
    let prevKey = ''
    let prevBucketId = -1
    for (let i = 0; i < list.length; i++) {
      const [key, sig] = list[i]
      const bucketId = sigToBucketId.get(sig)!
      let lcp = 0
      const minLen = Math.min(prevKey.length, key.length)
      while (lcp < minLen && prevKey.charCodeAt(lcp) === key.charCodeAt(lcp)) {
        lcp++
      }
      const suffixLen = key.length - lcp
      const sameBucket = bucketId === prevBucketId ? 1 : 0
      const singleSuffix = suffixLen === 1 ? 1 : 0
      const header = lcp * 4 + singleSuffix * 2 + sameBucket
      writer.writeVarint(header)
      if (singleSuffix === 0) {
        writer.writeVarint(suffixLen)
      }
      for (let k = lcp; k < key.length; k++) {
        writer.writeVarint(key.charCodeAt(k))
      }
      if (sameBucket === 0) {
        writer.writeVarint(bucketId)
      }
      prevKey = key
      prevBucketId = bucketId
    }
  }

  return {
    bytes: writer.finish(),
    stats: {
      regCount: sortedReg.length,
      uniqueBucketsCount: sortedUniqueBuckets.length,
      totalKeysCount,
      rawExportParts,
      rawExportBytes,
    },
  }
}

export function decodeCompactLexicalIndex(bytes: Uint8Array): DecodedCompactLexicalIndex {
  if (bytes.length > 64 * 1024 * 1024) throw new Error('Lexical payload exceeds size limit')
  let pos = 0
  const readVarint = (): number => {
    let result = 0
    let scale = 1
    for (let index = 0; index < 5; index++) {
      if (pos >= bytes.length) throw new Error('Truncated lexical index')
      const byte = bytes[pos++]
      result += (byte & 127) * scale
      if (result > 0xffffffff) throw new Error('Invalid lexical integer')
      if (byte < 128) return result
      scale *= 128
    }
    throw new Error('Invalid lexical varint')
  }
  const count = () => {
    const value = readVarint()
    if (value > bytes.length - pos) throw new Error('Invalid lexical count')
    return value
  }

  // 1. Decode reg
  const regLen = count()
  const reg: number[] = []
  reg.length = regLen
  let prevReg = 0
  for (let i = 0; i < regLen; i++) {
    const delta = readVarint()
    prevReg = i === 0 ? delta : prevReg + delta + 1
    if (prevReg !== i) throw new Error('Invalid lexical document IDs')
    reg[i] = prevReg
  }

  // 2. Decode unique buckets once
  const uniqueBucketsCount = count()
  const uniqueBuckets: BucketList[] = []
  uniqueBuckets.length = uniqueBucketsCount
  for (let u = 0; u < uniqueBucketsCount; u++) {
    const mask = readVarint()
    if (mask > 511) throw new Error('Invalid lexical score buckets')
    if (mask === 0) {
      uniqueBuckets[u] = []
      continue
    }
    const maxBit = 31 - Math.clz32(mask)
    const buckets: BucketList = []
    buckets.length = maxBit + 1
    for (let b = 0; b <= maxBit; b++) {
      if ((mask & (1 << b)) === 0) {
        buckets[b] = null
      } else {
        const len = count()
        const ids: number[] = []
        ids.length = len
        let curId = readVarint()
        if (!len || curId >= regLen) throw new Error('Invalid lexical posting')
        ids[0] = curId
        for (let j = 1; j < len; j++) {
          curId += readVarint() + 1
          if (curId >= regLen) throw new Error('Invalid lexical posting')
          ids[j] = curId
        }
        buckets[b] = ids
      }
    }
    uniqueBuckets[u] = buckets
  }

  // 3. Decode per-field front-coded UTF-16 keys
  const fields: Record<LexicalField, FieldMapEntry[]> = {
    title: [],
    fileName: [],
    content: [],
    tags: [],
    aliases: [],
  }

  let totalKeysCount = 0
  for (let fIdx = 0; fIdx < LEXICAL_FIELDS.length; fIdx++) {
    const field = LEXICAL_FIELDS[fIdx]
    const keyCount = count()
    totalKeysCount += keyCount
    const entries: FieldMapEntry[] = []
    entries.length = keyCount
    let prevKey = ''
    let prevBuckets: BucketList = []

    for (let i = 0; i < keyCount; i++) {
      const header = readVarint()
      const sameBucket = header & 1
      const singleSuffix = (header >>> 1) & 1
      const lcp = header >>> 2
      if (lcp > prevKey.length || (i === 0 && sameBucket)) throw new Error('Invalid lexical prefix')

      const prefix = lcp === prevKey.length ? prevKey : lcp === 0 ? '' : prevKey.slice(0, lcp)

      let key: string
      if (singleSuffix === 1) {
        const code = readVarint()
        if (code > 65535) throw new Error('Invalid lexical character')
        key = prefix + (code < 256 ? CHAR_TABLE[code] : String.fromCharCode(code))
      } else {
        const suffixLen = count()
        if (suffixLen === 0) {
          key = prefix
        } else if (suffixLen === 2) {
          const c0 = readVarint()
          const c1 = readVarint()
          if (c0 > 65535 || c1 > 65535) throw new Error('Invalid lexical character')
          key = prefix + String.fromCharCode(c0, c1)
        } else {
          const codes: number[] = []
          codes.length = suffixLen
          for (let k = 0; k < suffixLen; k++) {
            codes[k] = readVarint()
            if (codes[k] > 65535) throw new Error('Invalid lexical character')
          }
          let suffix = ''
          for (let start = 0; start < codes.length; start += 4096)
            suffix += String.fromCharCode(...codes.slice(start, start + 4096))
          key = prefix + suffix
        }
      }

      if (sameBucket === 0) {
        const bucketId = readVarint()
        if (bucketId >= uniqueBuckets.length) throw new Error('Invalid lexical bucket')
        prevBuckets = uniqueBuckets[bucketId]
      }

      if (i && key <= prevKey) throw new Error('Invalid lexical key order')
      entries[i] = [key, prevBuckets]
      prevKey = key
    }

    fields[field] = entries
  }

  if (pos !== bytes.length) throw new Error('Trailing lexical bytes')
  return { reg, fields, uniqueBucketsCount, totalKeysCount }
}

// Bump when the encoder, fields, tokenizer or FlexSearch version changes.
export const lexicalPrebuiltKind = 'quartz-lexical-forward-v1'

export function importLexicalQuery(
  index: LexicalDocumentIndex,
  decoded: DecodedCompactLexicalIndex,
  spec: LexicalSearchSpec,
): void {
  const terms = encode(spec.query)
  for (const field of spec.index) {
    const entries = decoded.fields[field]
    const matches: FieldMapEntry[] = []
    for (const term of terms) {
      let low = 0
      let high = entries.length
      while (low < high) {
        const middle = (low + high) >>> 1
        if (entries[middle][0] < term) low = middle + 1
        else high = middle
      }
      if (entries[low]?.[0] === term) matches.push(entries[low])
    }
    // FlexSearch accepts parsed export arrays; its published TS signature lists only strings.
    Reflect.apply(index.import, index, [`${field}.1.map`, matches])
  }
}
