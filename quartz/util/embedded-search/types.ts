export type SearchScope =
  | 'file'
  | 'path'
  | 'content'
  | 'tag'
  | 'match-case'
  | 'ignore-case'
  | 'line'
  | 'block'
  | 'section'
  | 'task'
  | 'task-todo'
  | 'task-done'

export type QueryExpression =
  | { readonly kind: 'term'; readonly value: string }
  | { readonly kind: 'regex'; readonly source: string; readonly flags: string }
  | { readonly kind: 'and' | 'or'; readonly children: readonly QueryExpression[] }
  | { readonly kind: 'not'; readonly child: QueryExpression }
  | { readonly kind: 'scope'; readonly scope: SearchScope; readonly child: QueryExpression }
  | { readonly kind: 'property'; readonly key: QueryExpression; readonly value?: QueryExpression }

export type CompiledQuery =
  | { readonly kind: 'compiled'; readonly expression: QueryExpression }
  | { readonly kind: 'error'; readonly message: string }

export interface SearchTask {
  readonly text: string
  readonly checked: boolean
}

export interface SearchDocument {
  readonly slug: string
  readonly path: string
  readonly title: string
  readonly text: string
  readonly tags: readonly string[]
  readonly blocks: readonly string[]
  readonly sections: readonly string[]
  readonly tasks: readonly SearchTask[]
  readonly properties: Readonly<Record<string, unknown>>
}

export type SearchCorpus =
  | { readonly kind: 'corpus'; readonly documents: readonly SearchDocument[] }
  | { readonly kind: 'error'; readonly message: string }

export type SearchEvaluation =
  | {
      readonly kind: 'results'
      readonly documents: readonly SearchDocument[]
      readonly truncated: boolean
    }
  | { readonly kind: 'error'; readonly message: string }

declare module 'vfile' {
  interface DataMap {
    embeddedSearchQueries?: readonly string[]
    authoredProperties?: Readonly<Record<string, unknown>>
  }
}

declare module 'hast' {
  interface ElementData {
    embeddedSearch?: boolean
  }
}
