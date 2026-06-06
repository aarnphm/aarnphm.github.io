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
    addCleanup(fn: (...args: any[]) => void)
    stacked: import('./quartz/types/plugin').Notes
    stackedNotePayloadCache?: Map<
      string,
      import('./quartz/components/scripts/util').StackedNotePayload
    >
    quartzToast: import('./quartz/components/scripts/toast').Toast
    plausible: { (eventName: string, options: { props: { path: string } }): void }
    twttr: { ready(f: (twttr: any) => void): void }
    mermaid: typeof import('mermaid/dist/mermaid').default
    mapboxgl: any
    pdfjsLib: any
    quartzPdfEmbeds?: { mount(root?: ParentNode): void; cleanup(root?: ParentNode): void }
  }
}
