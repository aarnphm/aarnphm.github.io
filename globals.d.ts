export declare global {
  interface Document {
    addEventListener<K extends keyof CustomEventMap>(
      type: K,
      listener: (this: Document, ev: CustomEventMap[K]) => void,
    ): void
    removeEventListener<K extends keyof CustomEventMap>(
      type: K,
      listener: (this: Document, ev: CustomEventMap[K]) => void,
    ): void
    dispatchEvent<K extends keyof CustomEventMap>(ev: CustomEventMap[K] | UIEvent): boolean
  }

  interface Window {
    spaNavigate(url: URL, isBack: boolean = false)
    notifyNav(url: FullSlug)
    addCleanup(fn: () => void)
    quartzNavLifecycle?: { listening: boolean; controller?: AbortController }
    quartzEscapeHandlers?: import('./quartz/components/scripts/escape-handler').EscapeHandlerRegistry
    quartzRootLifecycles?: import('./quartz/components/scripts/root-lifecycle').RootLifecycleRegistry
    quartzSidePanelSessions?: WeakMap<HTMLElement, () => void>
    quartzSidePanelRequests?: WeakMap<HTMLElement, AbortController>
    quartzCanvas?: { cleanup(root?: ParentNode): void }
    stacked: import('./quartz/types/plugin').Notes
    stackedNotePayloadCache?: Map<string, import('./quartz/util/stacked-notes').StackedNotePayload>
    quartzToast: import('./quartz/components/scripts/toast').Toast
    plausible: { (eventName: string, options: { props: { path: string } }): void }
    twttr: { ready(f: (twttr: any) => void): void }
    mermaid: typeof import('mermaid/dist/mermaid').default
    mapboxgl: any
    pdfjsLib: any
    quartzPdfEmbeds?: {
      mount(root?: ParentNode): void
      cleanup(root?: ParentNode): void
      preload(src?: string): Promise<void>
    }
    quartzTriathlon?: {
      dayCard(
        date: string,
        detailPath: string,
        extras?: { location?: string; event?: string; weightLbs?: number },
      ): Promise<HTMLElement | null>
    }
  }
}
