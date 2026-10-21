export function isSearchToggle(event: KeyboardEvent): boolean {
  return (
    !event.isComposing &&
    !event.altKey &&
    !event.shiftKey &&
    (event.metaKey || event.ctrlKey) &&
    (event.key.toLowerCase() === 'k' || event.key === '/')
  )
}

// Option+Shift+S types a different character on macOS (event.key is 'Í'), so match the physical key.
export function isSearchModeToggle(event: KeyboardEvent): boolean {
  return (
    !event.isComposing &&
    event.altKey &&
    event.shiftKey &&
    !event.metaKey &&
    !event.ctrlKey &&
    event.code === 'KeyS'
  )
}

export function searchModeShortcutLabel(mac: boolean): string {
  return mac ? '⌥⇧S' : 'Alt+Shift+S'
}
