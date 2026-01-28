import { ValidDateType } from "./components/Date"
import { ValidLocale } from "./i18n"
import { QuartzComponent } from "./types/component"
import { PluginTypes } from "./types/plugin"
import { Theme } from "./util/theme"

export type Analytics =
  | null
  | { provider: "plausible"; host?: string }
  | { provider: "google"; tagId: string }
  | { provider: "umami"; websiteId: string; host?: string }
  | { provider: "goatcounter"; websiteId: string; host?: string; scriptSrc?: string }
  | { provider: "posthog"; apiKey: string; host?: string }
  | { provider: "tinylytics"; siteId: string }
  | { provider: "cabin"; host?: string }

type DType = "fp16" | "fp32"

type SemanticIndexOptions = {
  /** Enable semantic search (default: true) */
  enable: boolean
  /** HuggingFace model ID for embeddings (e.g., "intfloat/multilingual-e5-large") */
  model: string
  /**
   * Ahead-of-time embedding generation (default: false)
   * - true: expect pre-generated embeddings in public/embeddings/
   * - false: run embed_build.py during build to generate embeddings
   * Requires: uv installed (https://docs.astral.sh/uv/)
   */
  aot: boolean
  /** Embedding dimension size (must match model output, e.g., 1024 for Qwen3-Embedding-0.6B) */
  dims: number
  /** Precision for stored vectors (fp16: smaller files, fp32: higher precision) */
  dtype: DType
  /** Number of vectors per shard file (default: 1024, higher = fewer files but larger downloads) */
  shardSizeRows: number
  /**
   * HNSW (Hierarchical Navigable Small World) graph parameters
   * - M: max connections per node (default: 16, higher = better recall but larger graph)
   * - efConstruction: candidate pool size during build (default: 200, higher = better quality but slower build)
   * - efSearch: search breadth (runtime only, not stored in manifest)
   */
  hnsw: { M?: number; efConstruction?: number; efSearch?: number }
  /**
   * Document chunking parameters (for long documents)
   * - chunkSize: max tokens per chunk (default: 512)
   * - chunkOverlap: overlap tokens between chunks (default: 128)
   * - noChunking: disable chunking, embed full documents (default: false)
   * - maxTokens: model's maximum context length in tokens (default: 512 for e5, 8192 for embeddinggemma)
   */
  chunking?: { chunkSize?: number; chunkOverlap?: number; noChunking?: boolean; maxTokens?: number }
  /**
   * vLLM server configuration (for remote embedding generation)
   * - useVllm: use vLLM API instead of local sentence-transformers (default: false)
   * - vllmUrl: vLLM server URL (default: from VLLM_URL env or http://127.0.0.1:8000/v1/embeddings)
   * - concurrency: concurrent requests to vLLM (default: 8)
   * - batchSize: batch size per request (default: 64)
   */
  vllm?: { useVllm?: boolean; vllmUrl?: string; concurrency?: number; batchSize?: number }
}

export interface GlobalConfiguration {
  pageTitle: string
  pageTitleSuffix?: string
  /** Whether to enable single-page-app style rendering. this prevents flashes of unstyled content and improves smoothness of Quartz */
  enableSPA: boolean
  /** Whether to display Wikipedia-style popovers when hovering over links */
  enablePopovers: boolean
  /** Analytics mode */
  analytics: Analytics
  /** Glob patterns to not search */
  ignorePatterns: string[]
  /** Whether to use created, modified, or published as the default type of date */
  defaultDateType: ValidDateType
  /** Base URL to use for CNAME files, sitemaps, and RSS feeds that require an absolute URL.
   *   Quartz will avoid using this as much as possible and use relative URLs most of the time
   */
  baseUrl?: string
  theme: Theme
  /**
   * Allow to translate the date in the language of your choice.
   * Also used for UI translation (default: en-US)
   * Need to be formatted following BCP 47: https://en.wikipedia.org/wiki/IETF_language_tag
   * The first part is the language (en) and the second part is the script/region (US)
   * Language Codes: https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
   * Region Codes: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
   */
  locale: ValidLocale
  /** Semantic search configuration */
  semanticSearch?: Partial<SemanticIndexOptions>
}

export interface QuartzConfig {
  configuration: GlobalConfiguration
  plugins: PluginTypes
}

export interface FullPageLayout {
  head: QuartzComponent
  header: QuartzComponent[]
  beforeBody: QuartzComponent[]
  pageBody: QuartzComponent
  afterBody: QuartzComponent[]
  sidebar: QuartzComponent[]
  footer: QuartzComponent
}

export type PageLayout = Pick<FullPageLayout, "beforeBody" | "sidebar">
export type SharedLayout = Pick<FullPageLayout, "head" | "header" | "footer" | "afterBody">

export const customMacros = {
  "\\argmin": "\\mathop{\\operatorname{arg\\,min}}\\limits",
  "\\argmax": "\\mathop{\\operatorname{arg\\,max}}\\limits",
  "\\upgamma": "\\mathit{\\gamma}",
  "\\upphi": "\\mathit{\\phi}",
  "\\upeta": "\\mathit{\\eta}",
  "\\upbeta": "\\mathit{\\beta}",
  "\\upalpha": "\\mathit{\\alpha}",
  "\\uptheta": "\\mathit{\\theta}",
  "\\abs": "\\left\\lvert #1 \\right\\rvert",
  // KaTeX does not support tabular/multicolumn. Provide safe fallbacks.
  // This macro drops alignment specifiers and yields only the cell content.
  // IMPORTANT: when spanning >1 columns, add explicit '&'s in source rows.
  "\\multicolumn": "#3",
  // Text micro symbol compatibility
  "\\textmu": "\\mu",
}

export const katexOptions = { strict: true, throwOnError: true }
