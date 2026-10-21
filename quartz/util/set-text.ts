// Per-frame text updates rewrite the existing text node. Replacing children (`textContent =`) sends
// every `:has()` rule on the page through invalidation, about 30 ms of style recalc per write on a
// triathlon day page.
export const setText = (node: Element, value: string): void => {
  const first = node.firstChild
  if (first instanceof Text && first === node.lastChild) {
    if (first.data !== value) first.data = value
  } else node.textContent = value
}
