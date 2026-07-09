export function debounce<Args extends unknown[]>(
  callback: (...args: Args) => void,
  delay: number,
): ((...args: Args) => void) & { cancel: () => void } {
  let timeoutId: ReturnType<typeof setTimeout> | undefined
  function debounced(...args: Args): void {
    if (timeoutId !== undefined) clearTimeout(timeoutId)
    timeoutId = setTimeout(() => callback(...args), delay)
  }
  debounced.cancel = () => {
    if (timeoutId === undefined) return
    clearTimeout(timeoutId)
    timeoutId = undefined
  }
  return debounced
}
