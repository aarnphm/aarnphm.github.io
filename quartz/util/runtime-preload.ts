function isIPadOS(): boolean {
  if (typeof navigator === 'undefined') return false
  const ua = navigator.userAgent
  if (/iPad/.test(ua)) return true
  return navigator.maxTouchPoints > 1 && /Macintosh/.test(ua)
}

export function supportsEagerRuntimePreload(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
  if (window.matchMedia('(prefers-reduced-data: reduce)').matches) return false
  return window.matchMedia('(pointer: fine)').matches || isIPadOS()
}
