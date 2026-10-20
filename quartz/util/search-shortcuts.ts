export function isSearchToggle(event: KeyboardEvent): boolean {
  return (
    !event.isComposing &&
    !event.altKey &&
    !event.shiftKey &&
    (event.metaKey || event.ctrlKey) &&
    (event.key.toLowerCase() === 'k' || event.key === '/')
  )
}

export function isSearchModeToggle(event: KeyboardEvent): boolean {
  return (
    !event.isComposing &&
    !event.altKey &&
    !event.metaKey &&
    !event.ctrlKey &&
    event.shiftKey &&
    event.key.toLowerCase() === 's'
  )
}
