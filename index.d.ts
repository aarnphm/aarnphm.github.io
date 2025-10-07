declare module "*.scss" {
  const content: string
  export = content
}

// dom custom event
interface CustomEventMap {
  prenav: CustomEvent<{}>
  nav: CustomEvent<{ url: FullSlug }>
  themechange: CustomEvent<{ theme: "light" | "dark" }>
  toast: CustomEvent<import("./quartz/components/scripts/toast").ToastEventDetail>
}

type ContentIndex = Record<FullSlug, ContentDetails>
declare const fetchData: Promise<ContentIndex>
declare const semanticCfg: import("./quartz/cfg").GlobalConfiguration["semanticSearch"]

// base view metadata for dropdown navigation
interface BaseViewMeta {
  name: string
  type: "table" | "list" | "gallery" | "board" | "calendar"
  slug: FullSlug
}

interface BaseMeta {
  baseSlug: FullSlug
  currentView: string
  allViews: BaseViewMeta[]
}
