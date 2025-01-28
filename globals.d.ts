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
    dispatchEvent<K extends keyof CustomEventMap>(ev: CustomEventMap[K] | UIEvent): void
  }

  interface Window {
    spaNavigate(url: URL, isBack: boolean = false)
    addCleanup(fn: (...args: any[]) => void)
    stacked: import("./quartz/plugins/types").Notes
    plausible: {
      (eventName: string, options: { props: { path: string } }): void
    }
    twttr: {
      ready(f: (twttr: any) => void): void
    }
    mermaid: typeof import("mermaid/dist/mermaid").default
  }
}
