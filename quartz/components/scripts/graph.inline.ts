import type { ContentDetails } from "../../plugins/emitters/contentIndex"
import {
  SimulationNodeDatum,
  SimulationLinkDatum,
  Simulation,
  forceSimulation,
  forceManyBody,
  forceCenter,
  forceLink,
  forceCollide,
  forceRadial,
  zoomIdentity,
  select,
  drag,
  zoom,
} from "d3"
import { Text, Graphics, Application, Container, Circle } from "pixi.js"
import { Group as TweenGroup, Tween as Tweened } from "@tweenjs/tween.js"
import { registerEscapeHandler, removeAllChildren } from "./util"
import { FullSlug, SimpleSlug, getFullSlug, resolveRelative, simplifySlug } from "../../util/path"
import { D3Config } from "../Graph"

type GraphicsInfo = {
  color: string
  gfx: Graphics
  alpha: number
  active: boolean
}

type NodeData = {
  id: SimpleSlug
  text: string
  tags: string[]
} & SimulationNodeDatum

type SimpleLinkData = {
  source: SimpleSlug
  target: SimpleSlug
}

type LinkData = {
  source: NodeData
  target: NodeData
} & SimulationLinkDatum<NodeData>

type NodeRenderData = GraphicsInfo & {
  simulationData: NodeData
  label: Text
}

type LinkRenderData = {
  simulationData: LinkData
  color: string
  alpha: number
  active: boolean
}

const localStorageKey = "graph-visited"
function getVisited(): Set<SimpleSlug> {
  return new Set(JSON.parse(localStorage.getItem(localStorageKey) ?? "[]"))
}

function addToVisited(slug: SimpleSlug) {
  const visited = getVisited()
  visited.add(slug)
  localStorage.setItem(localStorageKey, JSON.stringify([...visited]))
}

type TweenNode = {
  update: (time: number) => void
  stop: () => void
}

// workaround for pixijs webgpu issue: https://github.com/pixijs/pixijs/issues/11389
async function determineGraphicsAPI(): Promise<"webgpu" | "webgl"> {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null)
  const device = adapter && (await adapter.requestDevice().catch(() => null))
  if (!device) {
    return "webgl"
  }

  const canvas = document.createElement("canvas")
  const gl =
    (canvas.getContext("webgl2") as WebGL2RenderingContext | null) ??
    (canvas.getContext("webgl") as WebGLRenderingContext | null)

  if (!gl) return "webgl"

  const webglMaxTextures = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
  const webgpuMaxTextures = device.limits.maxSampledTexturesPerShaderStage

  return webglMaxTextures === webgpuMaxTextures ? "webgpu" : "webgl"
}

async function renderGraph(graph: HTMLElement, fullSlug: FullSlug) {
  const slug = simplifySlug(fullSlug)
  const visited = getVisited()
  removeAllChildren(graph)

  let {
    drag: enableDrag,
    zoom: enableZoom,
    depth,
    scale,
    opacityScale,
    repelForce,
    centerForce,
    linkDistance,
    fontSize,
    removeTags,
    showTags,
    focusOnHover,
    enableRadial,
  } = JSON.parse(graph.dataset["cfg"]!) as D3Config

  const data: Map<SimpleSlug, ContentDetails> = new Map(
    Object.entries<ContentDetails>(await fetchData).map(([k, v]) => [
      simplifySlug(k as FullSlug),
      v,
    ]),
  )
  const links: SimpleLinkData[] = []
  const tags = new Set<SimpleSlug>()
  const validLinks = new Set(data.keys())
  const adjacency = new Map<SimpleSlug, Set<SimpleSlug>>()

  const addAdjacent = (source: SimpleSlug, target: SimpleSlug) => {
    const existing = adjacency.get(source)
    if (existing) {
      existing.add(target)
      return
    }
    adjacency.set(source, new Set([target]))
  }

  const addLink = (source: SimpleSlug, target: SimpleSlug) => {
    links.push({ source, target })
    addAdjacent(source, target)
    addAdjacent(target, source)
  }

  const tweens = new Map<string, TweenNode>()
  for (const [source, details] of data.entries()) {
    const outgoing = details.links ?? []

    for (const dest of outgoing) {
      if (validLinks.has(dest)) {
        addLink(source, dest)
      }
    }

    if (showTags) {
      const localTags = details.tags
        .filter((tag) => !removeTags.includes(tag))
        .map((tag) => simplifySlug(("tags/" + tag) as FullSlug))

      for (const tag of localTags) {
        tags.add(tag)
      }

      for (const tag of localTags) {
        addLink(source, tag)
      }
    }
  }

  const neighbourhood = new Set<SimpleSlug>()
  const wl: (SimpleSlug | "__SENTINEL")[] = [slug, "__SENTINEL"]
  if (depth >= 0) {
    while (depth >= 0 && wl.length > 0) {
      const cur = wl.shift()!
      if (cur === "__SENTINEL") {
        depth--
        wl.push("__SENTINEL")
      } else if (!neighbourhood.has(cur)) {
        neighbourhood.add(cur)
        const neighbors = adjacency.get(cur)
        if (neighbors) {
          for (const next of neighbors) {
            if (!neighbourhood.has(next)) {
              wl.push(next)
            }
          }
        }
      }
    }
  } else {
    validLinks.forEach((id) => neighbourhood.add(id))
    if (showTags) {
      for (const tag of tags) {
        neighbourhood.add(tag)
      }
    }
  }

  const nodes = [...neighbourhood].map((url) => {
    const text = url.startsWith("tags/") ? "#" + url.substring(5) : (data.get(url)?.title ?? url)
    return {
      id: url,
      text,
      tags: data.get(url)?.tags ?? [],
    }
  })
  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  const filteredLinks = links.filter((l) => neighbourhood.has(l.source) && neighbourhood.has(l.target))
  const graphData: { nodes: NodeData[]; links: LinkData[] } = {
    nodes,
    links: filteredLinks.map((l) => ({
      source: nodeById.get(l.source)!,
      target: nodeById.get(l.target)!,
    })),
  }

  const nodeDegree = new Map<SimpleSlug, number>()
  for (const link of graphData.links) {
    nodeDegree.set(link.source.id, (nodeDegree.get(link.source.id) ?? 0) + 1)
    nodeDegree.set(link.target.id, (nodeDegree.get(link.target.id) ?? 0) + 1)
  }

  const nodeRadiusById = new Map<SimpleSlug, number>()
  for (const node of graphData.nodes) {
    const degree = nodeDegree.get(node.id) ?? 0
    nodeRadiusById.set(node.id, 4 + Math.log(degree + 1) * 2)
  }

  const width = graph.offsetWidth
  const height = Math.max(graph.offsetHeight, 250)

  const simulation: Simulation<NodeData, LinkData> = forceSimulation<NodeData>(graphData.nodes)
    .force("charge", forceManyBody().strength(-100 * repelForce))
    .force("center", forceCenter().strength(centerForce))
    .force("link", forceLink(graphData.links).distance(linkDistance))
    .force("collide", forceCollide<NodeData>((n) => nodeRadius(n)).iterations(3))

  if (enableRadial)
    simulation.force("radial", forceRadial((Math.min(width, height) / 2) * 0.8).strength(0.2))

  const cssVars = [
    "--secondary",
    "--tertiary",
    "--gray",
    "--light",
    "--lightgray",
    "--dark",
    "--darkgray",
    "--bodyFont",
    "--foam",
  ] as const
  const computedStyleMap = cssVars.reduce(
    (acc, key) => {
      acc[key] = getComputedStyle(document.documentElement).getPropertyValue(key)
      return acc
    },
    {} as Record<(typeof cssVars)[number], string>,
  )

  const color = (d: NodeData) => {
    const isCurrent = d.id === slug
    if (isCurrent) {
      return computedStyleMap["--secondary"]
    } else if (visited.has(d.id) || d.id.startsWith("tags/")) {
      return computedStyleMap["--tertiary"]
    } else {
      return computedStyleMap["--gray"]
    }
  }

  function nodeRadius(d: NodeData) {
    return nodeRadiusById.get(d.id) ?? 4
  }

  let hoveredNodeId: string | null = null
  let hoveredNeighbours: Set<string> = new Set()
  const linkRenderData: LinkRenderData[] = []
  const nodeRenderData: NodeRenderData[] = []
  function updateHoverInfo(newHoveredId: string | null) {
    hoveredNodeId = newHoveredId

    if (newHoveredId === null) {
      hoveredNeighbours = new Set()
      for (const n of nodeRenderData) {
        n.active = false
      }

      for (const l of linkRenderData) {
        l.active = false
      }
    } else {
      hoveredNeighbours = new Set()
      for (const l of linkRenderData) {
        const linkData = l.simulationData
        if (linkData.source.id === newHoveredId || linkData.target.id === newHoveredId) {
          hoveredNeighbours.add(linkData.source.id)
          hoveredNeighbours.add(linkData.target.id)
        }

        l.active = linkData.source.id === newHoveredId || linkData.target.id === newHoveredId
      }

      for (const n of nodeRenderData) {
        n.active = hoveredNeighbours.has(n.simulationData.id)
      }
    }
  }

  let dragStartTime = 0
  let dragging = false
  let animationFrame: number | null = null
  let stopAnimation = false
  let layoutDirty = true
  let linksDirty = true
  let renderDirty = true
  let tweenUntil = 0
  let linkTweenUntil = 0

  const scheduleRender = () => {
    if (stopAnimation || animationFrame !== null) return
    animationFrame = requestAnimationFrame(renderFrame)
  }

  const extendTween = (duration: number, affectsLinks: boolean) => {
    const now = performance.now()
    const endAt = now + duration
    tweenUntil = Math.max(tweenUntil, endAt)
    if (affectsLinks) {
      linkTweenUntil = Math.max(linkTweenUntil, endAt)
    }
    renderDirty = true
    scheduleRender()
  }

  const markLayoutDirty = () => {
    layoutDirty = true
    scheduleRender()
  }

  const markLinksDirty = () => {
    linksDirty = true
    scheduleRender()
  }

  const markRenderDirty = () => {
    renderDirty = true
    scheduleRender()
  }

  function renderLinks() {
    tweens.get("link")?.stop()
    const tweenGroup = new TweenGroup()

    for (const l of linkRenderData) {
      let alpha = 1

      if (hoveredNodeId) {
        alpha = l.active ? 1 : 0.2
      }

      l.color = l.active ? computedStyleMap["--secondary"] : computedStyleMap["--lightgray"]
      tweenGroup.add(new Tweened<LinkRenderData>(l).to({ alpha }, 200))
    }

    tweenGroup.getAll().forEach((tw) => tw.start())
    tweens.set("link", {
      update: tweenGroup.update.bind(tweenGroup),
      stop() {
        tweenGroup.getAll().forEach((tw) => tw.stop())
      },
    })
    extendTween(220, true)
    markLinksDirty()
  }

  const defaultScale = 1 / scale
  const activeScale = defaultScale * 1.1

  function renderLabels() {
    tweens.get("label")?.stop()
    const tweenGroup = new TweenGroup()

    for (const n of nodeRenderData) {
      const nodeId = n.simulationData.id
      const isCurrentlyHover = hoveredNodeId === nodeId
      const isConnected = hoveredNeighbours.has(nodeId)
      const scale =
        isCurrentlyHover || isConnected
          ? { x: activeScale, y: activeScale }
          : { x: defaultScale, y: defaultScale }

      const targetAlpha = isCurrentlyHover ? 1 : isConnected ? 0.8 : 0

      tweenGroup.add(new Tweened<Text>(n.label).to({ alpha: targetAlpha, scale }, 100))
    }

    tweenGroup.getAll().forEach((tw) => tw.start())
    tweens.set("label", {
      update: tweenGroup.update.bind(tweenGroup),
      stop() {
        tweenGroup.getAll().forEach((tw) => tw.stop())
      },
    })
    extendTween(120, false)
  }

  function renderNodes() {
    tweens.get("hover")?.stop()

    const tweenGroup = new TweenGroup()
    for (const n of nodeRenderData) {
      let alpha = 1

      if (hoveredNodeId !== null && focusOnHover) {
        alpha = n.active ? 1 : 0.2
      }

      tweenGroup.add(new Tweened<Graphics>(n.gfx, tweenGroup).to({ alpha }, 200))
    }

    tweenGroup.getAll().forEach((tw) => tw.start())
    tweens.set("hover", {
      update: tweenGroup.update.bind(tweenGroup),
      stop() {
        tweenGroup.getAll().forEach((tw) => tw.stop())
      },
    })
    extendTween(220, false)
  }

  function renderPixiFromD3() {
    renderNodes()
    renderLinks()
    renderLabels()
  }

  tweens.forEach((tween) => tween.stop())
  tweens.clear()

  const app = new Application()
  await app.init({
    width,
    height,
    antialias: true,
    autoStart: false,
    autoDensity: true,
    backgroundAlpha: 0,
    preference: await determineGraphicsAPI(),
    resolution: window.devicePixelRatio,
    eventMode: "static",
  })
  graph.appendChild(app.canvas)

  const stage = app.stage
  stage.interactive = false

  const labelsContainer = new Container<Text>({ zIndex: 3, isRenderGroup: true })
  const nodesContainer = new Container<Graphics>({ zIndex: 2, isRenderGroup: true })
  const linkContainer = new Container<Graphics>({ zIndex: 1, isRenderGroup: true })
  const linksGfx = new Graphics({ interactive: false, eventMode: "none" })
  linkContainer.addChild(linksGfx)
  stage.addChild(nodesContainer, labelsContainer, linkContainer)

  for (const simulationData of graphData.nodes) {
    const { text, id: nodeId } = simulationData

    const label = new Text({
      interactive: false,
      eventMode: "none",
      text,
      alpha: 0,
      anchor: { x: 0.5, y: 1.2 },
      style: {
        fontSize: fontSize * 15,
        fill: computedStyleMap["--dark"],
        fontFamily: computedStyleMap["--bodyFont"],
      },
      resolution: window.devicePixelRatio * 4,
    })
    label.scale.set(defaultScale)

    const isTagNode = nodeId.startsWith("tags/")
    const gfx = new Graphics({
      interactive: true,
      label: nodeId,
      eventMode: "static",
      hitArea: new Circle(0, 0, nodeRadius(simulationData)),
      cursor: "pointer",
    })
      .circle(0, 0, nodeRadius(simulationData))
      .fill({ color: isTagNode ? computedStyleMap["--foam"] : color(simulationData) })
      .on("pointerover", (e) => {
        updateHoverInfo(e.target.label)
        if (!dragging) renderPixiFromD3()
      })
      .on("pointerleave", () => {
        updateHoverInfo(null)
        if (!dragging) renderPixiFromD3()
      })

    if (isTagNode) {
      gfx.stroke({ width: 2, color: computedStyleMap["--tertiary"] })
    }

    nodesContainer.addChild(gfx)
    labelsContainer.addChild(label)

    nodeRenderData.push({
      simulationData,
      gfx,
      label,
      color: color(simulationData),
      alpha: 1,
      active: false,
    })
  }

  for (const l of graphData.links) {
    linkRenderData.push({
      simulationData: l,
      color: computedStyleMap["--lightgray"],
      alpha: 1,
      active: false,
    })
  }

  simulation.on("tick", () => {
    markLayoutDirty()
  })

  let currentTransform = zoomIdentity
  if (enableDrag) {
    select<HTMLCanvasElement, NodeData | undefined>(app.canvas).call(
      drag<HTMLCanvasElement, NodeData | undefined>()
        .container(() => app.canvas)
        .subject(() => graphData.nodes.find((n) => n.id === hoveredNodeId))
        .on("start", function dragstarted(event) {
          if (!event.active) simulation.alphaTarget(1).restart()
          event.subject.fx = event.subject.x
          event.subject.fy = event.subject.y
          event.subject.__initialDragPos = {
            x: event.subject.x,
            y: event.subject.y,
            fx: event.subject.fx,
            fy: event.subject.fy,
          }
          dragStartTime = Date.now()
          dragging = true
          markLayoutDirty()
        })
        .on("drag", function dragged(event) {
          const initPos = event.subject.__initialDragPos
          event.subject.fx = initPos.x + (event.x - initPos.x) / currentTransform.k
          event.subject.fy = initPos.y + (event.y - initPos.y) / currentTransform.k
          markLayoutDirty()
        })
        .on("end", function dragended(event) {
          if (!event.active) simulation.alphaTarget(0)
          event.subject.fx = null
          event.subject.fy = null
          dragging = false
          markLayoutDirty()

          if (Date.now() - dragStartTime < 500) {
            window.spaNavigate(
              new URL(resolveRelative(fullSlug, event.subject.id), window.location.toString()),
            )
          }
        }),
    )
  } else {
    for (const node of nodeRenderData) {
      node.gfx.on("click", () => {
        window.spaNavigate(
          new URL(resolveRelative(fullSlug, node.simulationData.id), window.location.toString()),
        )
        graph.classList.remove("active")
      })
    }
  }

  function handleZoomBehaviour(x: number, y: number, scale: number) {
    stage.scale.set(scale, scale)
    stage.position.set(x, y)

    const currentOpacityScale = scale * opacityScale
    let alpha = Math.max((currentOpacityScale - 1) / 3.75, 0)
    const activeNodes = nodeRenderData.filter((n) => n.active).flatMap((n) => n.label)

    for (const label of labelsContainer.children) {
      if (!activeNodes.includes(label)) {
        label.alpha = alpha
      }
    }
    markRenderDirty()
  }

  const initialScale = scale
  const initialX = width / 2 - (width / 2) * scale
  const initialY = height / 2 - (height / 2) * scale

  if (enableZoom) {
    const zoomBehaviour = zoom<HTMLCanvasElement, NodeData>()
      .extent([
        [0, 0],
        [width, height],
      ])
      .scaleExtent([0.25, 4])
      .on("zoom", ({ transform }) => {
        currentTransform = transform
        handleZoomBehaviour(transform.x, transform.y, transform.k)
      })

    const canvasSelection = select<HTMLCanvasElement, NodeData>(app.canvas).call(zoomBehaviour)
    const initialTransform = zoomIdentity.translate(initialX, initialY).scale(initialScale)
    canvasSelection.call(zoomBehaviour.transform, initialTransform)
  } else {
    handleZoomBehaviour(initialX, initialY, initialScale)
  }

  function renderFrame(time: number) {
    animationFrame = null
    if (stopAnimation) return

    if (layoutDirty) {
      for (const n of nodeRenderData) {
        const { x, y } = n.simulationData
        if (x == null || y == null) continue
        n.gfx.position.set(x + width / 2, y + height / 2)
        n.label.position.set(x + width / 2, y + height / 2)
      }
      layoutDirty = false
      linksDirty = true
      renderDirty = true
    }

    const tweenActive = time < tweenUntil
    const linkTweenActive = time < linkTweenUntil

    if (tweenActive) {
      tweens.forEach((t) => t.update(time))
      renderDirty = true
      if (linkTweenActive) {
        linksDirty = true
      }
    }

    if (linksDirty) {
      linksGfx.clear()
      for (const l of linkRenderData) {
        const linkData = l.simulationData
        if (
          linkData.source.x == null ||
          linkData.source.y == null ||
          linkData.target.x == null ||
          linkData.target.y == null
        )
          continue
        linksGfx.moveTo(linkData.source.x + width / 2, linkData.source.y + height / 2)
        linksGfx
          .lineTo(linkData.target.x + width / 2, linkData.target.y + height / 2)
          .stroke({ alpha: l.alpha, width: 1, color: l.color })
      }
      linksDirty = false
      renderDirty = true
    }

    if (renderDirty) {
      app.renderer.render(stage)
      renderDirty = false
    }

    if (tweenActive || linkTweenActive) {
      scheduleRender()
    }
  }

  markLayoutDirty()
  return () => {
    stopAnimation = true
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame)
      animationFrame = null
    }
    simulation.stop()
    app.destroy()
  }
}

let globalGraphCleanups: (() => void)[] = []

function cleanupGlobalGraphs() {
  for (const cleanup of globalGraphCleanups) {
    cleanup()
  }
  globalGraphCleanups = []
}

document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const notes = document.getElementById("stacked-notes-container")
  if (notes?.classList.contains("active")) return

  const slug = e.detail.url

  addToVisited(simplifySlug(slug))

  const containers = [...document.getElementsByClassName("global-graph-outer")] as HTMLElement[]

  async function renderGlobalGraph() {
    const slug = getFullSlug(window)

    for (const container of containers) {
      container.classList.add("active")
      const graphContainer = container.querySelector<HTMLElement>(".global-graph-container")
      registerEscapeHandler(container, hideGlobalGraph)
      if (!graphContainer) continue
      globalGraphCleanups.push(await renderGraph(graphContainer, slug))
    }
  }

  function hideGlobalGraph() {
    cleanupGlobalGraphs()
    for (const container of containers) {
      container.classList.remove("active")
    }
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if ((e.key === ";" || e.key === "g") && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const anyGlobalGraphOpen = containers.some((container) =>
        container.classList.contains("active"),
      )
      anyGlobalGraphOpen ? hideGlobalGraph() : renderGlobalGraph()
    }
  }

  const containerIcons = document.getElementsByClassName("global-graph-icon")
  Array.from(containerIcons).forEach((icon) => {
    icon.addEventListener("click", renderGlobalGraph)
    window.addCleanup(() => icon.removeEventListener("click", renderGlobalGraph))
  })

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => {
    document.removeEventListener("keydown", shortcutHandler)
    cleanupGlobalGraphs()
  })
})
