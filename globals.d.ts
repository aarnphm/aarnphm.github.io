interface StackedManager {
  active: boolean
  async open(): Promise<void>
  async add(href: URL): Promise<void>
  async restore(slugs: string[]): Promise<void>

  close(slug: string): void
  destroy(): void
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
    spaNavigate(url: URL, isBack: boolean = false)
    addCleanup(fn: (...args: any[]) => void)
    stacked: StackedManager
  }
}
