interface StackedManager {
  container: HTMLElement
  main: HTMLElement // the scrollable div
  column: HTMLElement // the actual container for all stacks

  active: boolean
  destroy(): void
  getChain(): string

  async open(): Promise<boolean>
  async add(href: URL): Promise<boolean>
  async navigate(url: URL): Promise<boolean>
}

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
    __MERMAID_RENDERED__: boolean
    spaNavigate(url: URL, isBack: boolean = false)
    addCleanup(fn: (...args: any[]) => void)
    stacked: StackedManager
    plausible: {
      (eventName: string, options: {props: {path: string}}): void
    }
    twttr: {
      ready(f: (twttr: any) => void): void
      private _e: any[]
    }
    mermaid: typeof import("mermaid/dist/mermaid").default
  }
}
