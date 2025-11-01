import { registerEscapeHandler, removeAllChildren, fetchCanonical } from "./util"
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

// configure marked for safe inline rendering
marked.setOptions({ breaks: true, gfm: true })

// helper to parse markdown safely
function parseMarkdown(text: string): string {
  try {
    return marked.parse(text, { async: false }) as string
  } catch (error) {
    console.error("Failed to parse markdown:", error)
    return escapeHtml(text)
  }
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
  // merged at runtime from data-meta (server-provided)
  displayName?: string
  resolvedSlug?: string
  resolvedHref?: string
  description?: string
  content?: string
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

type NodeData = CanvasNode & {
  fx?: number
  fy?: number
}

type LinkData = CanvasEdge & {
  source: NodeData
  target: NodeData
}

async function renderCanvas(container: HTMLElement) {
  const dataAttr = container.getAttribute("data-canvas")
  const cfgAttr = container.getAttribute("data-cfg")
  const metaAttr = container.getAttribute("data-meta")

  if (!dataAttr) return

  try {
    const canvasData: CanvasData = JSON.parse(dataAttr)
    const cfg: CanvasConfig = cfgAttr ? JSON.parse(cfgAttr) : {}
    const metaMap: Record<string, any> = metaAttr ? JSON.parse(metaAttr) : {}

    if (!canvasData.nodes || canvasData.nodes.length === 0) {
      container.textContent = "Empty canvas"
      return
    }

    removeAllChildren(container)

    const width = container.clientWidth || 800
    const height = container.clientHeight || 600

    // create SVG
    const svg = select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height])

    // add dot pattern for background
    const defs = svg.append("defs")
    const pattern = defs
      .append("pattern")
      .attr("id", `dots-${Math.random().toString(36).substr(2, 9)}`)
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

    // add background rect with pattern
    svg
      .append("rect")
      .attr("width", "100%")
      .attr("height", "100%")
      .attr("fill", `url(#${pattern.attr("id")})`)

    // create container groups - order matters for z-index
    const g = svg.append("g")
    const groupNodeGroup = g.append("g").attr("class", "group-nodes")
    const edgeGroup = g.append("g").attr("class", "edges")
    const nodeGroup = g.append("g").attr("class", "nodes")

    // setup zoom
    let zoomBehavior: any = null
    if (cfg.zoom) {
      zoomBehavior = d3Zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 4])
        .filter((event: any) => {
          // only allow zoom on shift+scroll, cmd+scroll, or ctrl+scroll
          if (event.type === "wheel") {
            return event.shiftKey || event.metaKey || event.ctrlKey
          }
          // allow other zoom interactions (pinch, drag, etc)
          return !event.button
        })
        .on("zoom", (event) => {
          g.attr("transform", event.transform)
        })
      svg.call(zoomBehavior as any)
    }

    // prepare data
    const nodes: NodeData[] = canvasData.nodes.map((n) => {
      const meta = (metaMap && (metaMap as any)[n.id]) || {}
      return {
        ...n,
        // merge server-provided metadata without polluting raw JSON on disk
        displayName: meta.displayName ?? n.displayName,
        resolvedSlug: meta.slug ?? n.resolvedSlug,
        resolvedHref: meta.href ?? n.resolvedHref,
        description: meta.description ?? n.description,
        content: meta.content ?? n.content,
        fx: cfg.useManualPositions ? n.x : undefined,
        fy: cfg.useManualPositions ? n.y : undefined,
      }
    })

    const nodeMap = new Map<string, NodeData>()
    nodes.forEach((n) => nodeMap.set(n.id, n))

    // separate groups from regular nodes
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

    // create force simulation for regular nodes only (groups are static)
    let simulation: any = null
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

    // render group nodes (behind everything else)
    const groupNode = groupNodeGroup
      .selectAll("g.node")
      .data(groupNodes)
      .join("g")
      .attr("class", "node node-group")
      .attr("data-node-id", (d) => d.id)
      .attr("data-color", (d) => d.color || "")

    groupNode
      .append("rect")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8)
      .attr("ry", 8)

    groupNode
      .append("text")
      .attr("class", "node-group-label")
      .attr("x", (d) => d.width / 2)
      .attr("y", 24)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "600")
      .style("text-transform", "uppercase")
      .style("letter-spacing", "0.05em")
      .text((d) => d.label || "Group")

    // render edges
    const edge = edgeGroup.selectAll("g.edge").data(links).join("g").attr("class", "edge")

    edge
      .append("path")
      .attr("stroke", (d) => d.color || "var(--gray)")
      .attr("stroke-width", 2)
      .attr("fill", "none")
      .attr("marker-end", "url(#arrowhead)")

    // edge labels with background
    const edgeLabel = edge.filter((d) => d.label).append("g").attr("class", "edge-label-group")

    edgeLabel
      .append("rect")
      .attr("class", "edge-label-bg")
      .attr("fill", "var(--light)")
      .attr("stroke", "var(--gray)")
      .attr("stroke-width", 1)
      .attr("rx", 3)
      .attr("ry", 3)

    edgeLabel
      .append("text")
      .attr("class", "edge-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.3em")
      .text((d) => d.label || "")

    // add arrowhead marker
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

    // render regular nodes (not groups)
    const node = nodeGroup
      .selectAll("g.node")
      .data(regularNodes)
      .join("g")
      .attr("class", (d) => `node node-${d.type}`)
      .attr("data-node-id", (d) => d.id)
      .attr("data-color", (d) => d.color || "")

    // node backgrounds
    node
      .append("rect")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8)
      .attr("ry", 8)
      .attr("stroke", "var(--gray)")
      .attr("stroke-width", 1.5)

    // title box for file nodes
    const fileNodes = node.filter((d) => d.type === "file")

    fileNodes
      .append("text")
      .attr("class", "node-title-text")
      .attr("x", 6)
      .attr("y", -6)
      .attr("fill", "var(--gray)")
      .attr("font-size", "12px")
      .attr("font-weight", "bold")
      .text((d) => d.displayName || d.file || "")
      .each(function (d: any) {
        const maxWidth = d.width - 32
        const textEl = this as SVGTextElement
        let text = d.displayName || d.file || ""
        textEl.textContent = text

        // truncate if too long
        while (textEl.getComputedTextLength() > maxWidth && text.length > 0) {
          text = text.slice(0, -1)
          textEl.textContent = text + "..."
        }
      })

    // node content
    node
      .append("foreignObject")
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .append("xhtml:div")
      .attr("class", "node-content")
      .html((d) => {
        if (d.type === "text") {
          // parse markdown for text nodes
          return `<div class="node-text">${parseMarkdown(d.text || "")}</div>`
        } else if (d.type === "file") {
          return `<div class="node-file-content">${d.content || (d.description ? escapeHtml(d.description) : "")}</div>`
        } else if (d.type === "link") {
          return `<div class="node-link"><span class="link-icon">🔗</span> ${escapeHtml(
            d.url || "",
          )}</div>`
        }
        return ""
      })

    // add a hidden overlay anchor inside file nodes to reuse site-wide popover logic
    node
      .filter((d) => d.type === "file")
      .select(".node-content")
      .each(function (d: any) {
        const container = this as unknown as HTMLElement
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
        link.style.pointerEvents = "none" // don’t block dragging
        container.appendChild(link)
      })

    // make nodes clickable for file/link types
    node
      .filter((d) => d.type === "file" || d.type === "link")
      .style("cursor", "pointer")
      .on("click", (evt, d) => {
        if (d.type === "file") {
          // Alt+click: delegate to popover/side-panel behavior
          if ((evt as MouseEvent).altKey) {
            const link = (evt.currentTarget as SVGGElement).querySelector(
              ".node-content a.canvas-popover-link",
            ) as HTMLAnchorElement | null
            if (link) {
              link.dispatchEvent(
                new MouseEvent("click", { altKey: true, bubbles: true, cancelable: true }),
              )
              return
            }
          }
          // Prefer fully-resolved href (including anchors) if available
          const resolvedHref = d.resolvedHref
          const resolvedSlug = d.resolvedSlug
          if (resolvedHref) {
            window.location.href = `/${resolvedHref}`
          } else if (resolvedSlug) {
            window.location.href = `/${resolvedSlug}`
          } else if (d.file) {
            // fallback to old behavior
            const htmlPath = d.file.replace(/\.md$/, "")
            window.location.href = htmlPath
          }
        } else if (d.type === "link" && d.url) {
          window.open(d.url, "_blank", "noopener,noreferrer")
        }
      })

    // setup dragging
    if (cfg.drag) {
      const drag = d3Drag<SVGGElement, NodeData>()
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

      node.call(drag as any)
    }

    // update positions
    function updatePositions() {
      // update regular nodes
      node.attr("transform", (d) => `translate(${d.x - d.width / 2},${d.y - d.height / 2})`)

      // update group nodes
      groupNode.attr("transform", (d) => `translate(${d.x - d.width / 2},${d.y - d.height / 2})`)

      // update edges to connect to node boundaries with fluid cubic curves
      edge.select("path").attr("d", (d) => {
        // use JSON Canvas spec sides if specified, otherwise auto-determine
        const p1 = getNodeEdgePoint(d.source, d.fromSide, d.target.x, d.target.y)
        const p2 = getNodeEdgePoint(d.target, d.toSide, d.source.x, d.source.y)

        const dx = p2.x - p1.x
        const dy = p2.y - p1.y
        const dist = Math.sqrt(dx * dx + dy * dy)

        // create fluid cubic bezier curve with two control points
        // control point offset based on distance for natural flow
        const offset = Math.min(dist * 0.3, 150)

        // horizontal bias for smoother, more natural curves
        const cp1x = p1.x + offset * (dx / dist)
        const cp1y = p1.y + offset * 0.2

        const cp2x = p2.x - offset * (dx / dist)
        const cp2y = p2.y - offset * 0.2

        // cubic bezier: M start C cp1 cp2 end
        return `M ${p1.x} ${p1.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`
      })

      // update edge labels - position at bezier curve midpoint (t=0.5)
      edge.selectAll(".edge-label-group").attr("transform", (d) => {
        const p1 = getNodeEdgePoint(d.source, d.fromSide, d.target.x, d.target.y)
        const p2 = getNodeEdgePoint(d.target, d.toSide, d.source.x, d.source.y)

        const dx = p2.x - p1.x
        const dy = p2.y - p1.y
        const dist = Math.sqrt(dx * dx + dy * dy)

        const offset = Math.min(dist * 0.3, 150)

        const cp1x = p1.x + offset * (dx / dist)
        const cp1y = p1.y + offset * 0.2

        const cp2x = p2.x - offset * (dx / dist)
        const cp2y = p2.y - offset * 0.2

        // cubic bezier at t=0.5: B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
        // at t=0.5: B(0.5) = 0.125·P0 + 0.375·P1 + 0.375·P2 + 0.125·P3
        const mx = 0.125 * p1.x + 0.375 * cp1x + 0.375 * cp2x + 0.125 * p2.x
        const my = 0.125 * p1.y + 0.375 * cp1y + 0.375 * cp2y + 0.125 * p2.y

        return `translate(${mx},${my})`
      })

      // size background rectangles to fit text
      edge.selectAll(".edge-label-group").each(function () {
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
      // manual positions: just set them once
      updatePositions()
    }

    // size edge label backgrounds after initial render
    setTimeout(() => {
      edge.selectAll(".edge-label-group").each(function () {
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

    // Center / fit to view using bounds from data-canvas-bounds
    const boundsAttr = container.getAttribute("data-canvas-bounds")
    if (boundsAttr) {
      try {
        const b = JSON.parse(boundsAttr) as {
          minX: number
          minY: number
          maxX: number
          maxY: number
        }
        const contentW = Math.max(1, b.maxX - b.minX)
        const contentH = Math.max(1, b.maxY - b.minY)
        const padding = 40
        const kRaw = Math.min((width - padding) / contentW, (height - padding) / contentH)
        const k = Math.max(0.1, Math.min(4, kRaw))
        const cx = (b.minX + b.maxX) / 2
        const cy = (b.minY + b.maxY) / 2
        const tx = width / 2 - k * cx
        const ty = height / 2 - k * cy
        if (zoomBehavior) {
          svg.call(zoomBehavior.transform, zoomIdentity.translate(tx, ty).scale(k))
        } else {
          g.attr("transform", `translate(${tx},${ty}) scale(${k})`)
        }
      } catch {}
    }

    // show preview on hover
    if (cfg.showPreviewOnHover) {
      const tooltip = select("body")
        .append("div")
        .attr("class", "canvas-tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "var(--light)")
        .style("border", "1px solid var(--border)")
        .style("border-radius", "4px")
        .style("padding", "8px")
        .style("max-width", "300px")
        .style("z-index", "1000")
        .style("font-size", "0.9em")

      node
        .filter((d) => d.type === "file")
        .on("mouseenter", async (event, d) => {
          // show description or full popover
          if (d.description) {
            // simple tooltip with description
            tooltip.html(escapeHtml(d.description)).style("visibility", "visible")
          } else {
            // show a lightweight popover with the page's .popover-hint contents
            const pointer = event as MouseEvent
            const href = d.resolvedHref
              ? `/${d.resolvedHref}`
              : d.resolvedSlug
                ? `/${d.resolvedSlug}`
                : (d.file || "").replace(/\.md$/, "")

            // create popover shell
            const pop = document.createElement("div")
            pop.classList.add("popover", "canvas-popover")
            const inner = document.createElement("div")
            inner.classList.add("popover-inner")
            pop.appendChild(inner)
            document.body.appendChild(pop)

            // simple fixed positioning near the pointer
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
              // normalize relative links to point to the fetched page
              html.querySelectorAll("[href], [src]").forEach((el) => {
                const e = el as HTMLElement
                const attr = e.hasAttribute("href") ? "href" : e.hasAttribute("src") ? "src" : null
                if (!attr) return
                const val = e.getAttribute(attr)!
                if (!val) return
                try {
                  const rebased = new URL(val, targetUrl)
                  e.setAttribute(attr, rebased.pathname + rebased.hash)
                } catch {}
              })
              // rewrite ids to avoid collisions (same as popover.inline.ts)
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
                // if there was a hash, try to scroll to it inside the popover
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

            // cleanup on leave
            const current = event.currentTarget as SVGGElement
            const onLeave = () => {
              pop.remove()
              current.removeEventListener("mouseleave", onLeave)
            }
            current.addEventListener("mouseleave", onLeave)
          }
        })
        .on("mousemove", (event, d) => {
          // only move tooltip for simple tooltips, not for full popovers
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
              link.dispatchEvent(new MouseEvent("mouseleave", { bubbles: true }))
            }
          }
        })

      registerEscapeHandler(container, () => {
        tooltip.remove()
      })
    }
  } catch (error) {
    console.error("Failed to render canvas:", error)
    container.textContent = `Error rendering canvas: ${error}`
  }
}

function escapeHtml(text: string): string {
  const div = document.createElement("div")
  div.textContent = text
  return div.innerHTML
}

// get connection point on node border
// if side specified (JSON Canvas spec), use center of that side
// otherwise calculate intersection point with target direction
function getNodeEdgePoint(
  node: NodeData,
  side?: string,
  targetX?: number,
  targetY?: number,
): { x: number; y: number } {
  const cx = node.x
  const cy = node.y
  const hw = node.width / 2
  const hh = node.height / 2

  // if side is specified, use center of that side (JSON Canvas spec)
  if (side) {
    switch (side) {
      case "top":
        return { x: cx, y: cy - hh }
      case "right":
        return { x: cx + hw, y: cy }
      case "bottom":
        return { x: cx, y: cy + hh }
      case "left":
        return { x: cx - hw, y: cy }
    }
  }

  // fallback: calculate intersection with rectangle border
  if (targetX !== undefined && targetY !== undefined) {
    const dx = targetX - cx
    const dy = targetY - cy

    if (dx === 0 && dy === 0) return { x: cx, y: cy }

    // find intersection with rectangle using parametric form
    const tx = Math.abs(dx) > 0 ? hw / Math.abs(dx) : Infinity
    const ty = Math.abs(dy) > 0 ? hh / Math.abs(dy) : Infinity
    const t = Math.min(tx, ty)

    return {
      x: cx + dx * t,
      y: cy + dy * t,
    }
  }

  return { x: cx, y: cy }
}

// initialize all canvases on the page
document.addEventListener("nav", () => {
  const selectors = [".canvas-container", ".canvas-embed-container"]
  const canvasContainers = document.querySelectorAll<HTMLElement>(selectors.join(","))
  canvasContainers.forEach((container) => {
    renderCanvas(container)
  })
})
