export interface TwitterWidgets {
  widgets: { load(root: HTMLElement): void | Promise<unknown> }
}

export interface TwitterLoader {
  ready(callback: (twitter: TwitterWidgets) => void): void
  _e?: ((twitter: TwitterWidgets) => void)[]
}

export function mountTwitterEmbeds(root: HTMLElement): void {
  if (!root.querySelector('blockquote.twitter-tweet')) return
  if (!window.twttr) {
    const callbacks: ((twitter: TwitterWidgets) => void)[] = []
    window.twttr = {
      _e: callbacks,
      ready(callback) {
        callbacks.push(callback)
      },
    }
  }
  if (!document.getElementById('twitter-wjs')) {
    const script = document.createElement('script')
    script.id = 'twitter-wjs'
    script.src = 'https://platform.twitter.com/widgets.js'
    script.async = true
    script.dataset.persist = 'true'
    document.head.appendChild(script)
  }
  window.twttr.ready(twitter => {
    if (root.isConnected) void twitter.widgets.load(root)
  })
}
