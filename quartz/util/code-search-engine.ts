import { readCodeManifest, readCodeShard } from './code-search-format'
import { codeMatch, codeUnitSelected, literalRanges, rankCodeMatches } from './code-search-match'
import {
  CODE_SEARCH_MANIFEST,
  type CodeMatch,
  type CodeQuery,
  type CodeSearchManifest,
  type CodeSearchProgress,
  type CodeSearchResult,
  type StructuralSource,
} from './code-search-types'

/** Worker-owned generation and source cache; a failed generation gets one fresh-manifest retry. */
export class CodeSearchEngine {
  private manifest: CodeSearchManifest | undefined
  private sources = new Map<string, string>()
  private validatedAt = 0
  constructor(
    private readonly manifestUrl: string = CODE_SEARCH_MANIFEST,
    private readonly manifestMaxAgeMs = 30_000,
  ) {}

  private async refresh(signal: AbortSignal, force = false): Promise<CodeSearchManifest> {
    signal.throwIfAborted()
    if (!force && this.manifest && performance.now() - this.validatedAt < this.manifestMaxAgeMs)
      return this.manifest
    const response = await fetch(this.manifestUrl, { signal, cache: 'no-cache' })
    if (!response.ok) throw new Error(`Code search manifest unavailable (${response.status})`)
    const data = readCodeManifest(await response.json())
    signal.throwIfAborted()
    if (data.generation !== this.manifest?.generation) this.sources.clear()
    this.manifest = data
    this.validatedAt = performance.now()
    return data
  }

  async search(
    query: CodeQuery,
    signal: AbortSignal,
    progress?: (value: CodeSearchProgress) => void,
  ): Promise<CodeSearchResult> {
    const started = performance.now()
    let manifest = await this.refresh(signal)
    try {
      return await this.run(manifest, query, signal, started, progress)
    } catch (error) {
      signal.throwIfAborted()
      const previous = manifest.generation
      manifest = await this.refresh(signal, true)
      if (manifest.generation === previous) throw error
      return this.run(manifest, query, signal, started, progress)
    }
  }

  private async run(
    manifest: CodeSearchManifest,
    query: CodeQuery,
    signal: AbortSignal,
    started: number,
    progress?: (value: CodeSearchProgress) => void,
  ): Promise<CodeSearchResult> {
    const units = manifest.units.filter(unit => codeUnitSelected(unit, query))
    const needed = new Set(units.map(unit => `${unit.language}:${unit.sourceId}`))
    const totalUnits = units.length
    const report = (phase: CodeSearchProgress['phase'], scannedUnits: number) =>
      progress?.({ phase, scannedUnits, totalUnits })
    report('loading', 0)
    for (const shard of manifest.shards) {
      if (
        !shard.sourceIds.some(
          id =>
            needed.has(`${shard.language}:${id}`) && !this.sources.has(`${shard.language}:${id}`),
        )
      )
        continue
      signal.throwIfAborted()
      const base = new URL(this.manifestUrl, globalThis.location?.href)
      const url = new URL(shard.path, base)
      if (
        url.origin !== base.origin ||
        !url.pathname.startsWith(base.pathname.slice(0, base.pathname.lastIndexOf('/') + 1))
      )
        throw new Error('Invalid code shard path')
      const response = await fetch(url, { signal })
      if (!response.ok) throw new Error(`Code search shard unavailable (${response.status})`)
      const data = await readCodeShard(
        new Uint8Array(await response.arrayBuffer()),
        shard,
        manifest.generation,
      )
      signal.throwIfAborted()
      for (const source of data.sources)
        this.sources.set(`${data.language}:${source.id}`, source.text)
    }
    const matches: CodeMatch[] = []
    let unsupportedLanguages: string[] = []
    let parseErrors = 0
    if (query.mode === 'structural') {
      const { searchStructural } = await import('../workers/code-search-ast')
      signal.throwIfAborted()
      const groups = new Map<string, Map<string, StructuralSource>>()
      for (const unit of units) {
        const text = this.sources.get(`${unit.language}:${unit.sourceId}`)
        if (text === undefined) throw new Error('Invalid missing code source')
        const group = groups.get(unit.language) ?? new Map<string, StructuralSource>()
        group.set(unit.sourceId, { id: unit.sourceId, language: unit.language, text })
        groups.set(unit.language, group)
      }
      const structuralRanges = new Map<string, ReturnType<typeof literalRanges>>()
      let scannedUnits = 0
      // Equal source hashes can have different grammars. Keep those results distinct.
      for (const [language, group] of groups) {
        const count = units.filter(unit => unit.language === language).length
        const result = await searchStructural(
          [...group.values()],
          { ...query, language },
          {
            cancelled: () => signal.aborted,
            progress: (scanned, total) =>
              report(
                'structural',
                scannedUnits + (total ? Math.floor((scanned / total) * count) : 0),
              ),
          },
        )
        signal.throwIfAborted()
        unsupportedLanguages.push(...result.unsupportedLanguages)
        parseErrors += result.parseErrors
        for (const match of result.matches)
          structuralRanges.set(`${language}:${match.sourceId}`, match.ranges)
        scannedUnits += count
      }
      for (const unit of units) {
        const text = this.sources.get(`${unit.language}:${unit.sourceId}`)
        const ranges = structuralRanges.get(`${unit.language}:${unit.sourceId}`)
        if (text !== undefined && ranges?.length) matches.push(codeMatch(unit, text, ranges, query))
      }
    } else {
      const cachedRanges = new Map<string, ReturnType<typeof literalRanges>>()
      let sliceStarted = performance.now()
      for (let i = 0; i < units.length; i++) {
        signal.throwIfAborted()
        const unit = units[i]
        const text = this.sources.get(`${unit.language}:${unit.sourceId}`)
        if (text === undefined) throw new Error('Invalid missing code source')
        let ranges = cachedRanges.get(unit.sourceId)
        if (!ranges) {
          ranges = literalRanges(text, query.pattern, query.caseSensitive)
          cachedRanges.set(unit.sourceId, ranges)
        }
        if (!query.pattern || ranges.length) matches.push(codeMatch(unit, text, ranges, query))
        if (performance.now() - sliceStarted >= 8) {
          report('searching', i + 1)
          await new Promise<void>(resolve => setTimeout(resolve, 0))
          sliceStarted = performance.now()
        }
      }
    }
    signal.throwIfAborted()
    report(query.mode === 'structural' ? 'structural' : 'searching', totalUnits)
    return {
      generation: manifest.generation,
      mode: query.mode,
      matches: rankCodeMatches(matches),
      totalMatches: matches.reduce((sum, match) => sum + match.ranges.length, 0),
      scannedUnits: totalUnits,
      totalUnits,
      unsupportedLanguages,
      parseErrors,
      complete: true,
      elapsedMs: performance.now() - started,
    }
  }
}
