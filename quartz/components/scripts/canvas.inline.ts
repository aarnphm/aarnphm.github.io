import type { DragBehavior } from "d3-drag"
import type { Simulation } from "d3-force"
import type { ZoomBehavior } from "d3-zoom"
import {
  forceSimulation,
  forceManyBody,
  forceCenter,
  forceLink,
  forceCollide,
  select,
  drag as d3Drag,
  zoom as d3Zoom,
  zoomIdentity,
} from "d3"
import { marked } from "marked"
import { normalizeRelativeURLs } from "../../util/path"
import { registerEscapeHandler, removeAllChildren, fetchCanonical } from "./util"

marked.setOptions({ breaks: true, gfm: true })

const EDGE_LABEL_LINE_HEIGHT = 1.2
const SYNTHETIC_MOUSELEAVE_FLAG = Symbol("canvasSyntheticMouseleave")

type SyntheticMouseEvent = MouseEvent & { [SYNTHETIC_MOUSELEAVE_FLAG]?: boolean }

function renderTextNodeContent(node: NodeData): string {
  const html = marked.parse(node.resolvedText ?? node.text ?? "", { async: false }) as string

  if (!node.wikilinks || node.wikilinks.length === 0) {
    return `<div class="node-text">${html}</div>`
  }

  const doc = htmlParser.parseFromString(html, "text/html")
  const anchors = Array.from(doc.body.querySelectorAll("a"))
  const remaining = [...node.wikilinks]

  anchors.forEach((anchor) => {
    const href = anchor.getAttribute("href") ?? ""
    const idx = remaining.findIndex((link) => link.resolvedHref === href)
    const match = idx >= 0 ? remaining.splice(idx, 1)[0] : undefined

    if (!match) {
      return
    }

    if (match.resolvedHref) {
      anchor.classList.add("internal")
      if (match.resolvedSlug) {
        const slug = match.resolvedSlug.startsWith("/")
          ? match.resolvedSlug
          : `/${match.resolvedSlug}`
        anchor.dataset.slug = slug
      }
    }

    if (match.missing) {
      anchor.classList.add("is-missing")
    }
  })

  normalizeRelativeURLs(doc, window.location.href)

  return `<div class="node-text">${doc.body.innerHTML}</div>`
}

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

type CanvasMeta = {
  slug?: string
  href?: string
  displayName?: string
  description?: string
  content?: string
  resolvedText?: string
  wikilinks?: CanvasResolvedWikilink[]
}

interface CanvasNode {
  id: string
  type: "text" | "file" | "link" | "group"
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

interface CanvasEdge {
  id: string
  fromNode: string
  toNode: string
  fromSide?: string
  toSide?: string
  label?: string
  color?: string
}

interface CanvasData {
  nodes: CanvasNode[]
  edges: CanvasEdge[]
}

interface CanvasConfig {
  drag: boolean
  zoom: boolean
  forceStrength: number
  linkDistance: number
  collisionRadius: number
  useManualPositions: boolean
  showInlineContent: boolean
  showPreviewOnHover: boolean
  previewMaxLength: number
}

type CanvasBounds = { minX: number; minY: number; maxX: number; maxY: number }

type NodeData = CanvasNode & { fx?: number; fy?: number }

type LinkData = CanvasEdge & { source: NodeData; target: NodeData }

const jsonFetchCache = new Map<string, Promise<unknown>>()
const filePreviewCache = new Map<string, Promise<string | null>>()
const htmlParser = new DOMParser()

function resolveToAbsoluteUrl(source: string): string {
  return new URL(source.trim(), window.location.href).toString()
}

async function fetchJson<T>(source: string): Promise<T> {
  const absoluteUrl = resolveToAbsoluteUrl(source)

  if (!jsonFetchCache.has(absoluteUrl)) {
    jsonFetchCache.set(
      absoluteUrl,
      fetch(absoluteUrl, { credentials: "same-origin" }).then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load canvas data (${response.status})`)
        }
        return response.json()
      }),
    )
  }

  return jsonFetchCache.get(absoluteUrl)! as Promise<T>
}

async function renderCanvas(container: HTMLElement) {
  const canvasUrl = container.getAttribute("data-canvas")?.trim()
  if (!canvasUrl) return

  const cfgAttr = container.getAttribute("data-cfg")
  const metaUrl = container.getAttribute("data-meta")?.trim()
  const isEmbed = container.getAttribute("data-embed") === "true"

  try {
    const [canvasData, metaMap] = await Promise.all([
      fetchJson<CanvasData>(canvasUrl),
      metaUrl
        ? fetchJson<Record<string, CanvasMeta>>(metaUrl)
        : Promise.resolve<Record<string, CanvasMeta>>({}),
    ])
    const cfg: CanvasConfig = cfgAttr ? JSON.parse(cfgAttr) : {}
    if (isEmbed) {
      cfg.drag = false
      cfg.zoom = false
    }

    if (!canvasData.nodes || canvasData.nodes.length === 0) {
      container.textContent = "Empty canvas"
      return
    }

    removeAllChildren(container)

    const width = container.clientWidth || 800
    const height = container.clientHeight || 600

    let toolbar: HTMLElement | null = null
    let helpModal: HTMLElement | null = null

    if (!isEmbed) {
      toolbar = document.createElement("div")
      toolbar.className = "canvas-controls"
      toolbar.innerHTML = `
        <div class="canvas-control-group">
          <button class="canvas-control-item" data-action="zoom-in" aria-label="Zoom in" title="Zoom in">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path><path d="M12 5v14"></path></svg>
          </button>
          <button class="canvas-control-item" data-action="zoom-reset" aria-label="Reset zoom" title="Reset zoom">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path></svg>
          </button>
          <button class="canvas-control-item" data-action="zoom-fit" aria-label="Zoom to fit" title="Zoom to fit">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3"></path><path d="M21 8V5a2 2 0 0 0-2-2h-3"></path><path d="M3 16v3a2 2 0 0 0 2 2h3"></path><path d="M16 21h3a2 2 0 0 0 2-2v-3"></path></svg>
          </button>
          <button class="canvas-control-item" data-action="zoom-out" aria-label="Zoom out" title="Zoom out">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path></svg>
          </button>
        </div>
        <div class="canvas-control-group">
          <button class="canvas-control-item" data-action="help" aria-label="Canvas help" title="Canvas help">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg>
          </button>
        </div>
      `
      container.appendChild(toolbar)

      helpModal = document.createElement("div")
      helpModal.className = "canvas-help-modal"
      helpModal.innerHTML = `
        <div class="canvas-help-backdrop"></div>
        <div class="canvas-help-content">
          <div class="canvas-help-header">
            <h2>Canvas help</h2>
            <button class="canvas-help-close" aria-label="Close">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>
            </button>
          </div>
          <div class="canvas-help-body">
            <div class="canvas-help-section">
              <h3>pan</h3>
              <div class="canvas-help-row">
                <span>Pan vertically</span>
                <span class="canvas-help-keys"><kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Pan horizontally</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>Scroll</kbd></span>
              </div>
            </div>
            <div class="canvas-help-section">
              <h3>zoom</h3>
              <div class="canvas-help-row">
                <span>Zoom</span>
                <span class="canvas-help-keys"><kbd>⌘/Ctrl</kbd> <kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Zoom to fit</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>1</kbd></span>
              </div>
            </div>
            <div class="canvas-help-section">
              <h3>navigation</h3>
              <div class="canvas-help-row">
                <span>Focus node</span>
                <span class="canvas-help-keys"><kbd>Click</kbd> on content</span>
              </div>
              <div class="canvas-help-row">
                <span>Open node</span>
                <span class="canvas-help-keys"><kbd>Click</kbd> outside content</span>
              </div>
              <div class="canvas-help-row">
                <span>Open from content</span>
                <span class="canvas-help-keys"><kbd>⌘/Ctrl</kbd> <kbd>Click</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Open in side panel</span>
                <span class="canvas-help-keys"><kbd>Alt</kbd> <kbd>Click</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Scroll node content</span>
                <span class="canvas-help-keys">Focus node, then <kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Defocus node</span>
                <span class="canvas-help-keys"><kbd>Esc</kbd></span>
              </div>
            </div>
          </div>
        </div>
      `
      container.appendChild(helpModal)
    }

    const svg = select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height])

    const defs = svg.append("defs")
    const pattern = defs
      .append("pattern")
      .attr("id", `dots-${Math.random().toString(36).substring(1, 10)}`)
      .attr("width", 20)
      .attr("height", 20)
      .attr("patternUnits", "userSpaceOnUse")

    pattern
      .append("circle")
      .attr("cx", 1)
      .attr("cy", 1)
      .attr("r", 1)
      .attr("fill", "var(--gray)")
      .attr("opacity", 0.3)

    svg
      .append("rect")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("fill", `url(#${pattern.attr("id")})`)

    const g = svg.append("g")
    const groupNodeGroup = g.append("g").attr("class", "group-nodes")
    const edgeGroup = g.append("g").attr("class", "edges")
    const edgeLabelGroup = g.append("g").attr("class", "edge-labels")
    const nodeGroup = g.append("g").attr("class", "nodes")

    let focusedNode: SVGGElement | null = null
    let isHelpOpen = false
    let bounds: CanvasBounds | null = null

    let zoomBehavior: ZoomBehavior<SVGSVGElement, unknown> | null = null
    let currentScale = 1
    let initialTransform = zoomIdentity
    if (cfg.zoom) {
      const zoom = d3Zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .filter((event) => {
          if (isHelpOpen) return false

          if (event.type === "wheel") {
            if (event.metaKey || event.ctrlKey) {
              return true
            }
            if (event.shiftKey) {
              return false
            }
            return false
          }
          return !event.button
        })
        .on("zoom", (event) => {
          g.attr("transform", event.transform)
          currentScale = event.transform.k

          if (currentScale < 0.5) {
            container.classList.add("canvas-skeleton-view")
          } else {
            container.classList.remove("canvas-skeleton-view")
          }
        })
      zoomBehavior = zoom
      svg.call(zoom)
    }

    const getBoundsTransform = (b: CanvasBounds) => {
      const contentW = Math.max(1, b.maxX - b.minX)
      const contentH = Math.max(1, b.maxY - b.minY)
      const padding = 40
      const kRaw = Math.min((width - padding) / contentW, (height - padding) / contentH)
      const k = Math.max(0.1, Math.min(4, kRaw))
      const cx = (b.minX + b.maxX) / 2
      const cy = (b.minY + b.maxY) / 2
      const tx = width / 2 - k * cx
      const ty = height / 2 - k * cy
      return { k, tx, ty }
    }

    const applyBoundsTransform = (b: CanvasBounds, animate: boolean) => {
      const { k, tx, ty } = getBoundsTransform(b)
      const transform = zoomIdentity.translate(tx, ty).scale(k)
      if (zoomBehavior) {
        const selection = animate ? svg.transition().duration(300) : svg
        selection.call(zoomBehavior.transform, transform)
      } else {
        g.attr("transform", `translate(${tx},${ty}) scale(${k})`)
      }
      return transform
    }

    container.addEventListener(
      "wheel",
      (event) => {
        if (isHelpOpen) return

        const target = event.target as HTMLElement
        if (focusedNode && target.closest(".node-content")) {
          return
        }

        if (!event.metaKey && !event.ctrlKey) {
          event.preventDefault()
          event.stopPropagation()

          const currentTransform = zoomBehavior
            ? ((svg.node() as SVGSVGElement & { __zoom?: typeof zoomIdentity }).__zoom ??
              zoomIdentity)
            : zoomIdentity

          const deltaX = event.deltaX
          const deltaY = event.deltaY
          const panSpeed = 1

          if (event.shiftKey) {
            const scrollDelta = deltaX !== 0 ? deltaX : deltaY
            const newTransform = currentTransform.translate(-scrollDelta * panSpeed, 0)
            if (zoomBehavior) {
              svg.call(zoomBehavior.transform, newTransform)
            } else {
              g.attr(
                "transform",
                `translate(${newTransform.x},${newTransform.y}) scale(${newTransform.k})`,
              )
            }
          } else {
            const newTransform = currentTransform.translate(0, -deltaY * panSpeed)
            if (zoomBehavior) {
              svg.call(zoomBehavior.transform, newTransform)
            } else {
              g.attr(
                "transform",
                `translate(${newTransform.x},${newTransform.y}) scale(${newTransform.k})`,
              )
            }
          }
        }
      },
      { passive: false },
    )

    const showHelp = () => {
      if (helpModal) {
        helpModal.classList.add("is-visible")
        isHelpOpen = true
      }
    }

    const hideHelp = () => {
      if (helpModal) {
        helpModal.classList.remove("is-visible")
        isHelpOpen = false
      }
    }

    if (helpModal) {
      const helpBackdrop = helpModal.querySelector(".canvas-help-backdrop")
      const helpClose = helpModal.querySelector(".canvas-help-close")

      if (helpBackdrop) {
        helpBackdrop.addEventListener("click", hideHelp)
      }

      if (helpClose) {
        helpClose.addEventListener("click", hideHelp)
      }
    }

    if (isEmbed) {
      const navigateTo = container.getAttribute("data-navigate-to")
      if (navigateTo) {
        const overlay = document.createElement("div")
        overlay.className = "canvas-embed-overlay"
        overlay.style.cssText =
          "position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1000; cursor: pointer;"
        overlay.setAttribute("aria-label", `view full canvas: ${navigateTo}`)
        overlay.setAttribute("role", "link")
        overlay.addEventListener("click", () => {
          const path = navigateTo.startsWith("/") ? navigateTo : `/${navigateTo}`
          window.spaNavigate(new URL(path, window.location.origin))
        })
        container.appendChild(overlay)
      }
    }

    if (toolbar) {
      toolbar.querySelectorAll("[data-action]").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation()
          const action = (btn as HTMLElement).getAttribute("data-action")

          if (action === "help") {
            showHelp()
            return
          }

          if (!zoomBehavior) return

          switch (action) {
            case "zoom-in":
              svg.transition().duration(300).call(zoomBehavior.scaleBy, 1.3)
              break
            case "zoom-out":
              svg.transition().duration(300).call(zoomBehavior.scaleBy, 0.7)
              break
            case "zoom-reset":
              svg.transition().duration(300).call(zoomBehavior.transform, initialTransform)
              break
            case "zoom-fit":
              if (bounds) {
                applyBoundsTransform(bounds, true)
              }
              break
          }
        })
      })
    }

    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        hideHelp()
        if (focusedNode) {
          focusedNode.classList.remove("is-focused")
          focusedNode = null
        }
      }
      if (event.shiftKey && event.key === "1") {
        event.preventDefault()
        if (bounds && zoomBehavior) {
          applyBoundsTransform(bounds, true)
        }
      }
    }

    document.addEventListener("keydown", handleKeydown)
    registerEscapeHandler(container, () => {
      hideHelp()
      if (focusedNode) {
        focusedNode.classList.remove("is-focused")
        focusedNode = null
      }
    })

    const nodes: NodeData[] = canvasData.nodes.map((n) => {
      const meta: CanvasMeta = metaMap[n.id] ?? {}
      return {
        ...n,
        displayName: meta.displayName ?? n.displayName,
        resolvedSlug: meta.slug ?? n.resolvedSlug,
        resolvedHref: meta.href ?? n.resolvedHref,
        description: meta.description ?? n.description,
        content: meta.content ?? n.content,
        resolvedText: meta.resolvedText ?? n.resolvedText,
        wikilinks: meta.wikilinks ?? n.wikilinks,
        fx: cfg.useManualPositions ? n.x : undefined,
        fy: cfg.useManualPositions ? n.y : undefined,
      }
    })

    const nodeMap = new Map<string, NodeData>()
    nodes.forEach((n) => nodeMap.set(n.id, n))

    const groupNodes = nodes.filter((n) => n.type === "group")
    const regularNodes = nodes.filter((n) => n.type !== "group")

    const links: LinkData[] = canvasData.edges
      .map((e) => {
        const source = nodeMap.get(e.fromNode)
        const target = nodeMap.get(e.toNode)
        if (!source || !target) return null
        return { ...e, source, target }
      })
      .filter((l): l is LinkData => l !== null)

    let simulation: Simulation<NodeData, LinkData> | null = null
    if (!cfg.useManualPositions) {
      simulation = forceSimulation(regularNodes)
        .force(
          "link",
          forceLink<NodeData, LinkData>(links)
            .id((d) => d.id)
            .distance(cfg.linkDistance || 150),
        )
        .force("charge", forceManyBody().strength(-cfg.forceStrength * 1000 || -300))
        .force("center", forceCenter(width / 2, height / 2))
        .force("collision", forceCollide().radius(cfg.collisionRadius || 50))
    }

    const groupNode = groupNodeGroup
      .selectAll("g.node")
      .data(groupNodes)
      .join("g")
      .attr("class", "node node-group")
      .attr("data-node-id", (d) => d.id)
      .attr("data-color", (d) => d.color || "")

    groupNode
      .append("rect")
      .attr("class", "node-bg")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8)
      .attr("ry", 8)

    groupNode
      .append("rect")
      .attr("class", "node-border-overlay")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("stroke", "var(--gray)")
      .attr("stroke-width", 1.5)
      .attr("fill", "none")
      .attr("pointer-events", "none")

    groupNode
      .append("text")
      .attr("class", "node-group-label")
      .attr("x", 12)
      .attr("y", -8)
      .attr("font-size", "2rem")
      .text((d) => d.label || "Group")
      .each(function (d: NodeData) {
        const maxWidth = d.width - 24
        const textEl = this as SVGTextElement
        let text = d.label || "Group"
        textEl.textContent = text

        while (textEl.getComputedTextLength() > maxWidth && text.length > 0) {
          text = text.slice(0, -1)
          textEl.textContent = text + "..."
        }
      })

    const edge = edgeGroup
      .selectAll<SVGGElement, LinkData>("g.edge")
      .data(links, (d) => d.id)
      .join(
        (enter) => {
          const edgeEnter = enter.append("g").attr("class", "edge")
          edgeEnter
            .append("path")
            .attr("stroke", (d) => d.color || "var(--gray)")
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr("marker-end", "url(#arrowhead)")
          return edgeEnter
        },
        (update) => update,
        (exit) => exit.remove(),
      )

    edge
      .select("path")
      .attr("stroke", (d) => d.color || "var(--gray)")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .attr("marker-end", "url(#arrowhead)")

    const labeledLinks = links.filter(
      (d): d is LinkData & { label: string } =>
        typeof d.label === "string" && d.label.trim().length > 0,
    )

    const edgeLabels = edgeLabelGroup
      .selectAll<SVGGElement, LinkData & { label: string }>("g.edge-label-group")
      .data(labeledLinks, (d) => d.id)
      .join(
        (enter) => {
          const labelGroup = enter
            .append("g")
            .attr("class", "edge-label-group")
            .style("pointer-events", "none")

          labelGroup
            .append("rect")
            .attr("class", "edge-label-bg")
            .attr("fill", "var(--light)")
            .attr("stroke", "var(--gray)")
            .attr("stroke-width", 1)
            .attr("rx", 4)
            .attr("ry", 4)
            .style("pointer-events", "none")

          labelGroup
            .append("text")
            .attr("class", "edge-label")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("pointer-events", "none")

          return labelGroup
        },
        (update) => update,
        (exit) => exit.remove(),
      )

    edgeLabels.select("text").each(function (d) {
      const textSelection = select(this as SVGTextElement)
      const normalized = d.label.split(/\r?\n/)
      const totalLines = normalized.length
      const offset = ((totalLines - 1) * EDGE_LABEL_LINE_HEIGHT) / 2
      const offsetStr = `${-offset.toFixed(3)}em`
      const lineStepStr = `${EDGE_LABEL_LINE_HEIGHT.toFixed(3)}em`
      textSelection.text(null)
      textSelection
        .selectAll("tspan")
        .data(normalized)
        .join("tspan")
        .attr("x", 0)
        .attr("dy", (_, i) => {
          if (totalLines === 1) {
            return "0em"
          }
          if (i === 0) {
            return offsetStr
          }
          return lineStepStr
        })
        .text((line) => (line.length === 0 ? "\u00A0" : line))
    })

    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 8)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "var(--gray)")

    const node = nodeGroup
      .selectAll<SVGGElement, NodeData>("g.node")
      .data(regularNodes)
      .join("g")
      .attr("class", (d) => `node node-${d.type}`)
      .attr("data-node-id", (d) => d.id)
      .attr("data-color", (d) => d.color || "")

    node
      .append("rect")
      .attr("class", "node-bg")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("stroke", "var(--gray)")
      .attr("stroke-width", 1.5)

    const fileNodes = node.filter((d) => d.type === "file")

    fileNodes
      .append("text")
      .attr("class", "node-title-text node-title-top")
      .attr("x", 12)
      .attr("y", -8)
      .attr("font-size", "12px")
      .text((d) => d.displayName || d.file || "")
      .each(function (d: NodeData) {
        const maxWidth = d.width - 32
        const textEl = this as SVGTextElement
        let text = d.displayName || d.file || ""
        textEl.textContent = text

        while (textEl.getComputedTextLength() > maxWidth && text.length > 0) {
          text = text.slice(0, -1)
          textEl.textContent = text + "..."
        }
      })

    fileNodes
      .append("text")
      .attr("class", "node-title-text node-title-center")
      .attr("x", (d) => d.width / 2)
      .attr("y", (d) => d.height / 2)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "600")
      .text((d) => d.displayName || d.file || "")
      .each(function (d: NodeData) {
        const maxWidth = d.width - 32
        const textEl = this as SVGTextElement
        let text = d.displayName || d.file || ""
        textEl.textContent = text

        while (textEl.getComputedTextLength() > maxWidth && text.length > 0) {
          text = text.slice(0, -1)
          textEl.textContent = text + "..."
        }
      })

    const textNodes = node.filter((d) => d.type === "text")

    textNodes
      .append("text")
      .attr("class", "node-title-text node-title-center")
      .attr("x", (d) => d.width / 2)
      .attr("y", (d) => d.height / 2)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .attr("font-size", "16px")
      .attr("font-weight", "600")

    textNodes.each(function (d: NodeData) {
      const host = select(this)
      const skeleton = host.append("g").attr("class", "node-skeleton")
      const paddingX = 12
      const paddingY = 16
      const availableWidth = Math.max(12, d.width - paddingX * 2)
      const lineHeight = 8
      const lineGap = 6
      const maxLines = Math.min(4, Math.floor((d.height - paddingY * 2) / (lineHeight + lineGap)))
      const lineCount = Math.max(1, maxLines)

      skeleton.attr("transform", `translate(${paddingX}, ${paddingY})`)

      for (let i = 0; i < lineCount; i++) {
        const lineWidth = availableWidth * (1 - i * 0.12)
        skeleton
          .append("rect")
          .attr("class", "node-skeleton-line")
          .attr("x", 0)
          .attr("y", i * (lineHeight + lineGap))
          .attr("width", Math.max(24, lineWidth))
          .attr("height", lineHeight)
          .attr("rx", 4)
          .attr("ry", 4)
      }
    })

    node
      .append("foreignObject")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .append("xhtml:div")
      .attr("class", "node-content")
      .html((d) => {
        if (d.type === "text") {
          return renderTextNodeContent(d)
        } else if (d.type === "file") {
          const hasDescription =
            typeof d.description === "string" && d.description.trim().length > 0
          const fallbackSource = hasDescription ? d.description! : "Loading preview..."
          const fallback = wrapTextHtml(fallbackSource)
          const placeholderAttr = hasDescription ? "" : ' data-placeholder="true"'
          return `<div class="node-file-content" data-loading="true"${placeholderAttr}>${fallback}</div>`
        } else if (d.type === "link") {
          return `<div class="node-link"><span class="link-icon">🔗</span> ${wrapTextHtml(
            d.url || "",
          )}</div>`
        }
        return ""
      })

    textNodes.selectAll(".node-skeleton").raise()

    node
      .filter((d) => d.type === "file")
      .select(".node-content")
      .each(function (d: NodeData) {
        const container = this as HTMLElement
        const link = document.createElement("a")
        link.className = "internal canvas-popover-link"
        const href = d.resolvedHref
          ? `/${d.resolvedHref}`
          : d.resolvedSlug
            ? `/${d.resolvedSlug}`
            : (d.file || "").replace(/\.md$/, "")
        link.href = href
        link.dataset.slug = d.resolvedSlug ? `/${d.resolvedSlug}` : href
        link.setAttribute("aria-hidden", "true")
        link.style.position = "absolute"
        link.style.left = "0"
        link.style.top = "0"
        link.style.width = "100%"
        link.style.height = "100%"
        link.style.opacity = "0"
        link.style.pointerEvents = "none"
        container.appendChild(link)
      })

    node
      .filter((d) => d.type === "file")
      .each(function (d: NodeData) {
        const host = this as SVGGElement
        const content = host.querySelector(".node-file-content") as HTMLElement | null
        if (!content) return
        hydrateFileNodeContent(content, d)
      })

    node
      .append("rect")
      .attr("class", "node-border-overlay")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("stroke", "var(--gray)")
      .attr("stroke-width", 1.5)
      .attr("fill", "none")
      .attr("pointer-events", "none")

    node.on("click", (evt, d) => {
      const clickEvent = evt as MouseEvent
      clickEvent.stopPropagation()
      const target = clickEvent.target as HTMLElement
      const currentNode = evt.currentTarget as SVGGElement

      if (target.closest("a:not(.canvas-popover-link)")) {
        return
      }

      if (focusedNode && focusedNode !== currentNode) {
        focusedNode.classList.remove("is-focused")
      }

      if (d.type === "file") {
        const clickedOnContent = target.closest(".node-content")

        if (clickEvent.altKey) {
          const link = currentNode.querySelector(
            ".node-content a.canvas-popover-link",
          ) as HTMLAnchorElement | null
          if (link) {
            link.dispatchEvent(
              new MouseEvent("click", { altKey: true, bubbles: true, cancelable: true }),
            )
            return
          }
        }

        if (clickedOnContent) {
          currentNode.classList.add("is-focused")
          focusedNode = currentNode

          if (clickEvent.metaKey || clickEvent.ctrlKey) {
            const resolvedHref = d.resolvedHref
            const resolvedSlug = d.resolvedSlug
            const navPath = resolvedHref || resolvedSlug || d.file?.replace(/\.md$/, "")
            if (navPath) {
              const fullPath = navPath.startsWith("/") ? navPath : `/${navPath}`
              window.spaNavigate(new URL(fullPath, window.location.origin))
            }
          }
          return
        }

        const resolvedHref = d.resolvedHref
        const resolvedSlug = d.resolvedSlug
        const navPath = resolvedHref || resolvedSlug || d.file?.replace(/\.md$/, "")
        if (navPath) {
          const fullPath = navPath.startsWith("/") ? navPath : `/${navPath}`
          window.spaNavigate(new URL(fullPath, window.location.origin))
        }
      } else if (d.type === "text") {
        currentNode.classList.add("is-focused")
        focusedNode = currentNode
      } else if (d.type === "link" && d.url) {
        window.open(d.url, "_blank", "noopener,noreferrer")
      }
    })

    svg.on("click", (evt) => {
      const target = evt.target as SVGElement
      if (
        target.tagName === "rect" ||
        target.tagName === "svg" ||
        target.classList.contains("canvas-background")
      ) {
        if (focusedNode) {
          focusedNode.classList.remove("is-focused")
          focusedNode = null
        }
      }
    })

    if (cfg.drag) {
      const drag: DragBehavior<SVGGElement, NodeData, unknown> = d3Drag<SVGGElement, NodeData>()
        .on("start", (event, d) => {
          if (!event.active && simulation) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on("drag", (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on("end", (event, d) => {
          if (!event.active && simulation) simulation.alphaTarget(0)
          if (!cfg.useManualPositions) {
            d.fx = undefined
            d.fy = undefined
          }
        })

      node.call(drag)
    }

    function updatePositions() {
      if (cfg.useManualPositions) {
        node.attr("transform", (d) => `translate(${d.x},${d.y})`)
      } else {
        node.attr("transform", (d) => `translate(${d.x - d.width / 2},${d.y - d.height / 2})`)
      }

      groupNode.attr("transform", (d) => `translate(${d.x},${d.y})`)

      edge.select("path").attr("d", (d) => {
        const sourceCenterX = d.source.x + d.source.width / 2
        const sourceCenterY = d.source.y + d.source.height / 2
        const targetCenterX = d.target.x + d.target.width / 2
        const targetCenterY = d.target.y + d.target.height / 2

        const p1 = getNodeEdgePoint(d.source, d.fromSide, targetCenterX, targetCenterY)
        const p2 = getNodeEdgePoint(d.target, d.toSide, sourceCenterX, sourceCenterY)

        const straightLength = 20
        const ext1 = getExtendedPoint(p1, straightLength)
        const ext2 = getExtendedPoint(p2, straightLength)

        const dx = ext2.x - ext1.x
        const dy = ext2.y - ext1.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        const controlDist = Math.min(dist * 0.4, 100)

        const cp1 = getControlPoint(ext1, p1.side, controlDist)
        const cp2 = getControlPoint(ext2, p2.side, controlDist)

        return `M ${p1.x} ${p1.y} L ${ext1.x} ${ext1.y} C ${cp1.x} ${cp1.y}, ${cp2.x} ${cp2.y}, ${ext2.x} ${ext2.y} L ${p2.x} ${p2.y}`
      })

      edgeLabels.attr("transform", (d) => {
        const sourceCenterX = d.source.x + d.source.width / 2
        const sourceCenterY = d.source.y + d.source.height / 2
        const targetCenterX = d.target.x + d.target.width / 2
        const targetCenterY = d.target.y + d.target.height / 2

        const p1 = getNodeEdgePoint(d.source, d.fromSide, targetCenterX, targetCenterY)
        const p2 = getNodeEdgePoint(d.target, d.toSide, sourceCenterX, sourceCenterY)

        const straightLength = 20
        const ext1 = getExtendedPoint(p1, straightLength)
        const ext2 = getExtendedPoint(p2, straightLength)

        const dx = ext2.x - ext1.x
        const dy = ext2.y - ext1.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        const controlDist = Math.min(dist * 0.4, 100)

        const cp1 = getControlPoint(ext1, p1.side, controlDist)
        const cp2 = getControlPoint(ext2, p2.side, controlDist)

        const t = 0.5
        const mt = 1 - t
        const mx =
          mt * mt * mt * ext1.x +
          3 * mt * mt * t * cp1.x +
          3 * mt * t * t * cp2.x +
          t * t * t * ext2.x
        const my =
          mt * mt * mt * ext1.y +
          3 * mt * mt * t * cp1.y +
          3 * mt * t * t * cp2.y +
          t * t * t * ext2.y

        return `translate(${mx},${my})`
      })

      edgeLabels.each(function () {
        const group = this as SVGGElement
        const text = group.querySelector("text") as SVGTextElement
        const bg = group.querySelector("rect") as SVGRectElement
        if (text && bg) {
          const bbox = text.getBBox()
          const padding = 4
          bg.setAttribute("x", String(bbox.x - padding))
          bg.setAttribute("y", String(bbox.y - padding))
          bg.setAttribute("width", String(bbox.width + padding * 2))
          bg.setAttribute("height", String(bbox.height + padding * 2))
        }
      })
    }

    if (simulation) {
      simulation.on("tick", updatePositions)
    } else {
      updatePositions()
    }

    setTimeout(() => {
      edgeLabels.each(function () {
        const group = this as SVGGElement
        const text = group.querySelector("text") as SVGTextElement
        const bg = group.querySelector("rect") as SVGRectElement
        if (text && bg) {
          const bbox = text.getBBox()
          const padding = 4
          bg.setAttribute("x", String(bbox.x - padding))
          bg.setAttribute("y", String(bbox.y - padding))
          bg.setAttribute("width", String(bbox.width + padding * 2))
          bg.setAttribute("height", String(bbox.height + padding * 2))
        }
      })
    }, 100)

    const boundsAttr = container.getAttribute("data-canvas-bounds")
    if (boundsAttr) {
      bounds = JSON.parse(boundsAttr)
    } else if (isEmbed) {
      let minX = Infinity
      let minY = Infinity
      let maxX = -Infinity
      let maxY = -Infinity

      for (const node of nodes) {
        const x1 = node.x
        const y1 = node.y
        const x2 = node.x + node.width
        const y2 = node.y + node.height

        minX = Math.min(minX, x1)
        minY = Math.min(minY, y1)
        maxX = Math.max(maxX, x2)
        maxY = Math.max(maxY, y2)
      }

      if (isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
        bounds = { minX, minY, maxX, maxY }
      }
    }

    if (bounds) {
      initialTransform = applyBoundsTransform(bounds, false)
    }

    if (cfg.showPreviewOnHover) {
      const tooltip = select("body")
        .append("div")
        .attr("class", "canvas-tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "var(--light)")
        .style("border", "1px solid var(--border)")
        .style("border-radius", "var(--radius-md)")
        .style("padding", "8px")
        .style("max-width", "300px")
        .style("z-index", "1000")
        .style("font-size", "0.9em")

      node
        .filter((d) => d.type === "file")
        .on("mouseenter", async (event, d) => {
          if (d.description) {
            tooltip.html(wrapTextHtml(d.description)).style("visibility", "visible")
          } else {
            const pointer = event as MouseEvent
            const href = d.resolvedHref
              ? `/${d.resolvedHref}`
              : d.resolvedSlug
                ? `/${d.resolvedSlug}`
                : (d.file || "").replace(/\.md$/, "")

            const pop = document.createElement("div")
            pop.classList.add("popover", "canvas-popover")
            const inner = document.createElement("div")
            inner.classList.add("popover-inner")
            pop.appendChild(inner)
            document.body.appendChild(pop)

            pop.style.position = "fixed"
            pop.style.left = `${pointer.clientX + 12}px`
            pop.style.top = `${pointer.clientY + 12}px`

            try {
              const url = new URL(href, window.location.toString())
              const targetUrl = new URL(url.toString())
              const hash = decodeURIComponent(targetUrl.hash)
              targetUrl.hash = ""
              targetUrl.search = ""
              const response = await fetchCanonical(targetUrl)
              const html = new DOMParser().parseFromString(await response.text(), "text/html")
              html.querySelectorAll("[href], [src]").forEach((el) => {
                const e = el as HTMLElement
                const attr = e.hasAttribute("href") ? "href" : e.hasAttribute("src") ? "src" : null
                if (!attr) return
                const val = e.getAttribute(attr)!
                if (!val) return
                const rebased = new URL(val, targetUrl)
                e.setAttribute(attr, rebased.pathname + rebased.hash)
              })
              html.querySelectorAll("[id]").forEach((el) => {
                const targetID = `popover-${el.id}`
                el.id = targetID
              })
              const elts = [
                ...(html.getElementsByClassName(
                  "popover-hint",
                ) as HTMLCollectionOf<HTMLDivElement>),
              ]
              if (elts.length > 0) {
                inner.append(...elts)
                if (hash) {
                  const targetAnchor = hash.startsWith("#popover")
                    ? hash
                    : `#popover-${hash.slice(1)}`
                  const heading = inner.querySelector(targetAnchor) as HTMLElement | null
                  if (heading) {
                    inner.scroll({ top: heading.offsetTop - 12 })
                  }
                }
                inner.classList.add("grid")
              }
            } catch (e) {
              console.error("canvas popover failed: ", e)
            }

            const current = event.currentTarget as SVGGElement
            const onLeave = () => {
              pop.remove()
              current.removeEventListener("mouseleave", onLeave)
            }
            current.addEventListener("mouseleave", onLeave)
          }
        })
        .on("mousemove", (event, d) => {
          if (d.type === "file" && d.description) {
            tooltip.style("top", `${event.pageY + 10}px`).style("left", `${event.pageX + 10}px`)
          }
        })
        .on("mouseleave", (event, d) => {
          tooltip.style("visibility", "hidden")
          if (d.type === "file") {
            const link = (event.currentTarget as SVGGElement).querySelector(
              ".node-content a.canvas-popover-link",
            ) as HTMLAnchorElement | null
            if (link) {
              const mouseleaveEvent = event as SyntheticMouseEvent
              if (mouseleaveEvent[SYNTHETIC_MOUSELEAVE_FLAG]) {
                return
              }

              const synthetic = new MouseEvent("mouseleave", {
                bubbles: true,
              }) as SyntheticMouseEvent
              synthetic[SYNTHETIC_MOUSELEAVE_FLAG] = true
              link.dispatchEvent(synthetic)
            }
          }
        })

      registerEscapeHandler(container, () => {
        tooltip.remove()
      })
    }
  } catch (error) {
    console.error("Failed to render canvas:", error)
    container.innerHTML =
      '<div class="canvas-error">Failed to load canvas data. Check the console for details.</div>'
  }
}

async function hydrateFileNodeContent(container: HTMLElement, node: NodeData): Promise<void> {
  container.dataset.loading = "true"

  try {
    const preview = await getFileNodePreview(node)
    if (!preview) {
      if (container.dataset.placeholder === "true" && container.isConnected) {
        container.textContent = "Preview unavailable"
        container.removeAttribute("data-placeholder")
      }
      return
    }

    if (!container.isConnected) return

    container.innerHTML = preview
    container.removeAttribute("data-placeholder")
  } finally {
    if (container.isConnected) {
      container.dataset.loading = "false"
    }
  }
}

async function getFileNodePreview(node: NodeData): Promise<string | null> {
  const targetUrl = resolveFileNodeUrl(node)
  if (!targetUrl) return null

  const cacheKey = targetUrl.toString()

  if (!filePreviewCache.has(cacheKey)) {
    filePreviewCache.set(
      cacheKey,
      (async () => {
        try {
          const response = await fetchCanonical(targetUrl)
          if (!response.ok) {
            return null
          }

          const contentType = response.headers.get("Content-Type") ?? ""
          if (!contentType.startsWith("text/html")) {
            return null
          }

          const markup = await response.text()
          const doc = htmlParser.parseFromString(markup, "text/html")
          normalizeRelativeURLs(doc, targetUrl)

          doc
            .querySelectorAll('section[class~="page-footer"],section[class~="page-header"]')
            .forEach((el) => el.remove())

          doc.querySelectorAll("[id]").forEach((el) => {
            const targetID = `canvas-${el.id}`
            el.id = targetID
          })

          const hints = Array.from(
            doc.getElementsByClassName("popover-hint") as HTMLCollectionOf<HTMLElement>,
          )

          if (hints.length > 0) {
            return hints
              .map(
                (hint) => cleanCanvasPreviewElement(hint.cloneNode(true) as HTMLElement).outerHTML,
              )
              .join("")
          }

          const article = doc.querySelector("article")
          if (!article) {
            return null
          }

          const clone = article.cloneNode(true) as HTMLElement
          cleanCanvasPreviewElement(clone)
          return clone.innerHTML
        } catch (error) {
          console.error("Canvas preview failed:", error)
          return null
        }
      })(),
    )
  }

  return filePreviewCache.get(cacheKey)!
}

function resolveFileNodeUrl(node: NodeData): URL | null {
  const base =
    (typeof node.resolvedHref === "string" && node.resolvedHref.trim().length > 0
      ? node.resolvedHref
      : null) ??
    (typeof node.resolvedSlug === "string" && node.resolvedSlug.trim().length > 0
      ? node.resolvedSlug
      : null) ??
    (typeof node.file === "string" && node.file.trim().length > 0
      ? node.file.replace(/\.md$/, "")
      : null)

  if (!base) return null

  const [rawPath] = base.split("#")
  const trimmed = rawPath.trim()
  if (trimmed.length === 0) return null

  const normalizedPath = trimmed.startsWith("/") ? trimmed : `/${trimmed}`
  return new URL(normalizedPath, window.location.origin)
}

function cleanCanvasPreviewElement<T extends HTMLElement>(element: T): T {
  const removable = element.querySelectorAll<HTMLElement>(
    "section[data-references], section[data-footnotes], [data-skip-preview], .telescopic-container",
  )
  removable.forEach((node) => node.remove())

  const iconSelector = 'svg[data-icon="github"], svg[data-icon="twitter"]'
  element.querySelectorAll<SVGElement>(iconSelector).forEach((node) => node.remove())
  return element
}

function wrapTextHtml(text: string): string {
  const div = document.createElement("div")
  div.textContent = text
  return div.innerHTML
}

function getNodeEdgePoint(
  node: NodeData,
  side?: string,
  targetX?: number,
  targetY?: number,
): { x: number; y: number; side: string } {
  const cx = node.x + node.width / 2
  const cy = node.y + node.height / 2
  const hw = node.width / 2
  const hh = node.height / 2

  if (side) {
    switch (side) {
      case "top":
        return { x: cx, y: cy - hh, side: "top" }
      case "right":
        return { x: cx + hw, y: cy, side: "right" }
      case "bottom":
        return { x: cx, y: cy + hh, side: "bottom" }
      case "left":
        return { x: cx - hw, y: cy, side: "left" }
    }
  }

  if (targetX !== undefined && targetY !== undefined) {
    const dx = targetX - cx
    const dy = targetY - cy

    if (dx === 0 && dy === 0) return { x: cx, y: cy, side: "right" }

    const tx = Math.abs(dx) > 0 ? hw / Math.abs(dx) : Infinity
    const ty = Math.abs(dy) > 0 ? hh / Math.abs(dy) : Infinity
    const t = Math.min(tx, ty)

    const x = cx + dx * t
    const y = cy + dy * t

    let determinedSide = "right"
    if (Math.abs(y - (cy - hh)) < 1) determinedSide = "top"
    else if (Math.abs(y - (cy + hh)) < 1) determinedSide = "bottom"
    else if (Math.abs(x - (cx - hw)) < 1) determinedSide = "left"
    else if (Math.abs(x - (cx + hw)) < 1) determinedSide = "right"

    return { x, y, side: determinedSide }
  }

  return { x: cx, y: cy, side: "right" }
}

function getExtendedPoint(point: { x: number; y: number; side: string }, length: number = 20) {
  switch (point.side) {
    case "top":
      return { x: point.x, y: point.y - length }
    case "right":
      return { x: point.x + length, y: point.y }
    case "bottom":
      return { x: point.x, y: point.y + length }
    case "left":
      return { x: point.x - length, y: point.y }
    default:
      return { x: point.x, y: point.y }
  }
}

function getControlPoint(extPoint: { x: number; y: number }, side: string, distance: number) {
  switch (side) {
    case "top":
      return { x: extPoint.x, y: extPoint.y - distance }
    case "right":
      return { x: extPoint.x + distance, y: extPoint.y }
    case "bottom":
      return { x: extPoint.x, y: extPoint.y + distance }
    case "left":
      return { x: extPoint.x - distance, y: extPoint.y }
    default:
      return { x: extPoint.x, y: extPoint.y }
  }
}

document.addEventListener("nav", () => {
  document
    .querySelectorAll<HTMLElement>([".canvas-container", ".canvas-embed-container"].join(","))
    .forEach((container) => {
      renderCanvas(container)
    })
})
