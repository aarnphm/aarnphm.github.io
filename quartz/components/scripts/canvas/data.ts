interface CanvasResolvedWikilink {
  raw: string
  target?: string
  anchor?: string
  alias?: string
  embed?: boolean
  resolvedSlug?: string
  resolvedHref?: string
  missing?: boolean
}

export type CanvasMeta = {
  slug?: string
  href?: string
  displayName?: string
  description?: string
  content?: string
  resolvedText?: string
  wikilinks?: CanvasResolvedWikilink[]
}

export interface CanvasNode {
  id: string
  type: 'text' | 'file' | 'link' | 'group'
  x: number
  y: number
  width: number
  height: number
  color?: string
  text?: string
  file?: string
  url?: string
  label?: string
  displayName?: string
  resolvedSlug?: string
  resolvedHref?: string
  description?: string
  content?: string
  resolvedText?: string
  wikilinks?: CanvasResolvedWikilink[]
}

export interface CanvasEdge {
  id: string
  fromNode: string
  toNode: string
  fromSide?: string
  toSide?: string
  label?: string
  color?: string
}

export interface CanvasData {
  nodes: CanvasNode[]
  edges: CanvasEdge[]
}

export interface CanvasConfig {
  zoom: boolean
}

export type CanvasBounds = { minX: number; minY: number; maxX: number; maxY: number }

export type NodeData = CanvasNode

export type LinkData = CanvasEdge & { source: NodeData; target: NodeData }

const jsonFetchCache = new Map<string, Promise<unknown>>()

function resolveToAbsoluteUrl(source: string): string {
  return new URL(source.trim(), window.location.href).toString()
}

export async function fetchJson<T>(source: string): Promise<T> {
  const absoluteUrl = resolveToAbsoluteUrl(source)
  let request = jsonFetchCache.get(absoluteUrl)
  if (!request) {
    request = fetch(absoluteUrl, { credentials: 'same-origin' }).then(response => {
      if (!response.ok) {
        throw new Error(`Failed to load canvas data (${response.status})`)
      }
      return response.json()
    })
    jsonFetchCache.set(absoluteUrl, request)
  }

  try {
    return (await request) as T
  } catch (error) {
    if (jsonFetchCache.get(absoluteUrl) === request) jsonFetchCache.delete(absoluteUrl)
    throw error
  }
}
