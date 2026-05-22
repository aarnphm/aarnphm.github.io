declare module '*.scss' {
  const content: string
  export = content
}

declare module '*.inline' {
  const content: string
  export default content
}

declare module '*.inline.ts' {
  const content: string
  export default content
}

declare module '*.inline.js' {
  const content: string
  export default content
}

declare module '*.html' {
  const content: string
  export default content
}

// dom custom event
interface CustomEventMap {
  prenav: CustomEvent<{}>
  nav: CustomEvent<{ url: FullSlug }>
  themechange: CustomEvent<{ theme: 'light' | 'dark' }>
  readermodechange: CustomEvent<{ mode: 'on' | 'off' }>
  petstoggle: CustomEvent<{ enabled?: boolean }>

  slidechange: CustomEvent<{}>
  toast: CustomEvent<import('./quartz/components/scripts/toast').ToastEventDetail>
  collapsibletoggle: CustomEvent<{ toggleId: string; isOpen: 'true' | 'false' }>
  commentauthorupdated: CustomEvent<{ oldAuthor: string; newAuthor: string }>
  contentdecrypted: CustomEvent<{ article: HTMLDivElement; content: HTMLDivElement }>
}

type ContentIndex = Record<FullSlug, ContentDetails>
declare const fetchData: Promise<ContentIndex>
declare const fetchSearchData: Promise<ContentIndex>
declare const semanticCfg: import('./quartz/cfg').GlobalConfiguration['semanticSearch']

// base view metadata for dropdown navigation
interface BaseViewMetadata {
  name: string
  type: 'table' | 'list' | 'gallery' | 'board' | 'calendar' | 'card' | 'cards' | 'map'
  slug: FullSlug
}

interface BaseMetadata {
  baseSlug: FullSlug
  currentView: string
  allViews: BaseViewMetadata[]
}
