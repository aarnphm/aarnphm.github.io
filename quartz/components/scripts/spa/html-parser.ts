let p: DOMParser

export function parseHtml(markup: string): Document {
  p = p || new DOMParser()
  return p.parseFromString(markup, 'text/html')
}
