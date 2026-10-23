import type { ImageSemantic } from '../image-semantic'
import type { ImageSimilarity } from '../image-similarity'
import type { SiteSemantic } from '../site-semantic'
import { FullSlug } from '../../../util/path'

export type Item = {
  id: number
  slug: FullSlug
  title: string
  content: string
  tags: string[]
  aliases: string[]
  target: string
  fileName: string
  protected?: boolean
  titleMatch?: boolean
}

export type SearchType = 'basic' | 'tags'
export type SearchMode = 'lexical' | 'semantic'

// Every search element reads and writes this state, and it lives across navigations.
export const shared: {
  searchMode: SearchMode
  currentSearchTerm: string
  rawSearchTerm: string
  semantic: SiteSemantic | null
  semanticReady: boolean
  semanticInitFailed: boolean
} = {
  searchMode: 'lexical',
  currentSearchTerm: '',
  rawSearchTerm: '',
  semantic: null,
  semanticReady: false,
  semanticInitFailed: false,
}

export type SimilarityResult = { item: Item; similarity: number }

// The image search engines of one search element.
export type SearchImage = {
  similarity: ImageSimilarity
  ids: ReadonlySet<string> | undefined
  semantic: ImageSemantic | null
  ready: Promise<ImageSemantic> | null
  active: boolean
}

// The state of one search element. setupSearch creates it; the phases below read and write it.
export type SearchSession = {
  searchElement: HTMLDivElement
  currentSlug: FullSlug
  container: HTMLElement
  searchButton: HTMLButtonElement
  searchBar: HTMLInputElement
  searchLayout: HTMLOutputElement
  searchSpace: HTMLFormElement
  image: SearchImage
  progressBar: HTMLDivElement
  progressResetTimer: number
  progressReasons: Set<'search' | 'preview'>
  data: ContentIndex | null
  idDataMap: FullSlug[]
  slugToIndex: Map<string, number>
  searchDataReady: Promise<ContentIndex> | null
  modeToggle: HTMLDivElement | null
  modeButtons: HTMLButtonElement[]
  semanticStatus: HTMLElement | null
  enablePreview: boolean
  previewVisible: boolean
  semanticBooting: Promise<boolean> | null
  searchSeq: number
  runSearchTimer: number | null
  requestedMode: SearchMode
  modeRequest: number
  preview: HTMLDivElement | undefined
  previewRequest: AbortController | undefined
  results: HTMLDivElement
  currentHover: HTMLElement | null
  previewToggle: HTMLButtonElement | null
  previousFocus: HTMLElement | null
  pressedKeys: Set<string>
  composing: boolean
  pendingNavigation: -1 | 1 | null
  pendingOpen: number | null
  pendingDisplay: { token: number; apply: () => void | Promise<void> } | null
  // Previews and result rows start an image similarity query through this callback.
  showSimilarImages: (seed: string) => Promise<void>
}
