import type {
  ReferenceResolution,
  ReferenceTargetDiagnostic,
  ReferenceSelector,
  ReferenceSettings,
} from './catalog-types'
import type { ReferenceRoute } from './route'

export interface ReferenceRequestRecord {
  readonly target: string
  readonly origin: string
  readonly fragments: readonly string[]
  readonly settings?: string
  readonly embed: boolean
  readonly decodeUri: boolean
  readonly implicitAlias: boolean
  readonly label?: string
  readonly authoredSpan?: { readonly start: number; readonly end: number }
}

export interface ReferenceDependency extends ReferenceRequestRecord {
  readonly status: ReferenceResolution['kind']
  readonly lookupKeys: readonly string[]
  readonly resolvedPath?: string
  readonly resolvedSlug?: string
  readonly route?: ReferenceRoute
  readonly selector?: ReferenceSelector
  readonly resolvedSettings?: ReferenceSettings
  readonly implicitLabel?: string
  readonly diagnostics: readonly ReferenceTargetDiagnostic[]
  readonly members?: readonly ReferenceDependency[]
}

declare module 'hast' {
  interface Data {
    referenceRequest?: ReferenceDependency
  }
}

declare module 'mdast' {
  interface Data {
    referenceRequest?: ReferenceRequestRecord
    resolvedReference?: ReferenceDependency
    referenceId?: string
  }
}

declare module 'vfile' {
  interface DataMap {
    referenceDependencies?: ReferenceDependency[]
    referenceResolutionSignature?: string
    referenceTransportPrefix?: string
  }
}
