import { parseJsonCanvas, serializeJcast } from "./parser"
import {
  JcastCanvas,
  JsonCanvas,
  JsonCanvasNode,
  JsonCanvasEdge,
  TextNodeOptions,
  FileNodeOptions,
  LinkNodeOptions,
  GroupNodeOptions,
  EdgeBuilderOptions,
} from "./types"

/**
 * Fluent builder for programmatically creating JSON Canvas files
 *
 * Example usage:
 * ```ts
 * const canvas = CanvasBuilder.create()
 *   .addTextNode({ x: 0, y: 0, width: 200, height: 100, text: "Hello" })
 *   .addFileNode({ x: 300, y: 0, width: 200, height: 100, file: "thoughts/LLMs.md" })
 *   .addEdge({ fromNode: node1.id, toNode: node2.id, label: "relates to" })
 *   .build()
 * ```
 */
export class CanvasBuilder {
  private nodes: JsonCanvasNode[] = []
  private edges: JsonCanvasEdge[] = []
  private idCounter = 0

  private constructor() {}

  /**
   * Create a new canvas builder
   */
  static create(): CanvasBuilder {
    return new CanvasBuilder()
  }

  /**
   * Load an existing canvas for modification
   */
  static fromCanvas(canvas: JsonCanvas | JcastCanvas): CanvasBuilder {
    const builder = new CanvasBuilder()

    if ("type" in canvas && canvas.type === "canvas") {
      // it's a jcast, serialize it back to JSON Canvas
      const jsonCanvas = serializeJcast(canvas as JcastCanvas)
      builder.nodes = [...jsonCanvas.nodes]
      builder.edges = [...jsonCanvas.edges]
    } else {
      // it's already JSON Canvas
      const jsonCanvas = canvas as JsonCanvas
      builder.nodes = [...jsonCanvas.nodes]
      builder.edges = [...jsonCanvas.edges]
    }

    // set id counter to max existing ID + 1
    const maxId = Math.max(
      ...builder.nodes.map((n) => parseInt(n.id) || 0),
      ...builder.edges.map((e) => parseInt(e.id) || 0),
      0,
    )
    builder.idCounter = maxId + 1

    return builder
  }

  /**
   * Generate a unique ID
   */
  private generateId(): string {
    return `generated-${this.idCounter++}`
  }

  /**
   * Add a text node
   */
  addTextNode(options: TextNodeOptions): CanvasBuilder {
    this.nodes.push({
      id: options.id || this.generateId(),
      type: "text",
      x: options.x,
      y: options.y,
      width: options.width,
      height: options.height,
      color: options.color,
      text: options.text,
    })
    return this
  }

  /**
   * Add a file node
   */
  addFileNode(options: FileNodeOptions): CanvasBuilder {
    this.nodes.push({
      id: options.id || this.generateId(),
      type: "file",
      x: options.x,
      y: options.y,
      width: options.width,
      height: options.height,
      color: options.color,
      file: options.file,
      subpath: options.subpath,
    })
    return this
  }

  /**
   * Add a link node
   */
  addLinkNode(options: LinkNodeOptions): CanvasBuilder {
    this.nodes.push({
      id: options.id || this.generateId(),
      type: "link",
      x: options.x,
      y: options.y,
      width: options.width,
      height: options.height,
      color: options.color,
      url: options.url,
    })
    return this
  }

  /**
   * Add a group node
   */
  addGroupNode(options: GroupNodeOptions): CanvasBuilder {
    this.nodes.push({
      id: options.id || this.generateId(),
      type: "group",
      x: options.x,
      y: options.y,
      width: options.width,
      height: options.height,
      color: options.color,
      label: options.label,
      background: options.background,
      backgroundStyle: options.backgroundStyle,
    })
    return this
  }

  /**
   * Add an edge between nodes
   */
  addEdge(options: EdgeBuilderOptions): CanvasBuilder {
    this.edges.push({
      id: options.id || this.generateId(),
      fromNode: options.fromNode,
      fromSide: options.fromSide,
      fromEnd: options.fromEnd,
      toNode: options.toNode,
      toSide: options.toSide,
      toEnd: options.toEnd,
      color: options.color,
      label: options.label,
    })
    return this
  }

  /**
   * Remove a node by ID
   */
  removeNode(nodeId: string): CanvasBuilder {
    this.nodes = this.nodes.filter((n) => n.id !== nodeId)
    // also remove edges connected to this node
    this.edges = this.edges.filter((e) => e.fromNode !== nodeId && e.toNode !== nodeId)
    return this
  }

  /**
   * Remove an edge by ID
   */
  removeEdge(edgeId: string): CanvasBuilder {
    this.edges = this.edges.filter((e) => e.id !== edgeId)
    return this
  }

  /**
   * Update a node
   */
  updateNode(nodeId: string, updates: Partial<JsonCanvasNode>): CanvasBuilder {
    const node = this.nodes.find((n) => n.id === nodeId)
    if (node) {
      Object.assign(node, updates)
    }
    return this
  }

  /**
   * Update an edge
   */
  updateEdge(edgeId: string, updates: Partial<JsonCanvasEdge>): CanvasBuilder {
    const edge = this.edges.find((e) => e.id === edgeId)
    if (edge) {
      Object.assign(edge, updates)
    }
    return this
  }

  /**
   * Filter nodes by predicate
   */
  filterNodes(predicate: (node: JsonCanvasNode) => boolean): CanvasBuilder {
    const removedIds = new Set(this.nodes.filter((n) => !predicate(n)).map((n) => n.id))
    this.nodes = this.nodes.filter(predicate)
    // remove edges connected to removed nodes
    this.edges = this.edges.filter((e) => !removedIds.has(e.fromNode) && !removedIds.has(e.toNode))
    return this
  }

  /**
   * Transform all nodes
   */
  mapNodes(transform: (node: JsonCanvasNode) => JsonCanvasNode): CanvasBuilder {
    this.nodes = this.nodes.map(transform)
    return this
  }

  /**
   * Get node by ID
   */
  getNode(nodeId: string): JsonCanvasNode | undefined {
    return this.nodes.find((n) => n.id === nodeId)
  }

  /**
   * Get edge by ID
   */
  getEdge(edgeId: string): JsonCanvasEdge | undefined {
    return this.edges.find((e) => e.id === edgeId)
  }

  /**
   * Build the final jcast AST
   */
  build(): JcastCanvas {
    return parseJsonCanvas({ nodes: this.nodes, edges: this.edges })
  }

  /**
   * Build as JSON Canvas (not AST)
   */
  buildJson(): JsonCanvas {
    return { nodes: [...this.nodes], edges: [...this.edges] }
  }

  /**
   * Build as JSON string
   */
  buildJsonString(pretty: boolean = true): string {
    return pretty ? JSON.stringify(this.buildJson(), null, 2) : JSON.stringify(this.buildJson())
  }
}

/**
 * Utility functions for common canvas operations
 */
export class CanvasUtils {
  /**
   * Merge multiple canvases into one
   */
  static merge(...canvases: JsonCanvas[]): JsonCanvas {
    const nodes: JsonCanvasNode[] = []
    const edges: JsonCanvasEdge[] = []
    const seenNodeIds = new Set<string>()
    const seenEdgeIds = new Set<string>()

    for (const canvas of canvases) {
      for (const node of canvas.nodes) {
        if (!seenNodeIds.has(node.id)) {
          nodes.push(node)
          seenNodeIds.add(node.id)
        }
      }
      for (const edge of canvas.edges) {
        if (!seenEdgeIds.has(edge.id)) {
          edges.push(edge)
          seenEdgeIds.add(edge.id)
        }
      }
    }

    return { nodes, edges }
  }

  /**
   * Create a subgraph containing only specified nodes
   */
  static subgraph(canvas: JsonCanvas, nodeIds: Set<string>): JsonCanvas {
    const nodes = canvas.nodes.filter((n) => nodeIds.has(n.id))
    const edges = canvas.edges.filter((e) => nodeIds.has(e.fromNode) && nodeIds.has(e.toNode))
    return { nodes, edges }
  }

  /**
   * Clone a canvas
   */
  static clone(canvas: JsonCanvas): JsonCanvas {
    return {
      nodes: canvas.nodes.map((n) => ({ ...n })),
      edges: canvas.edges.map((e) => ({ ...e })),
    }
  }

  /**
   * Transform node positions
   */
  static translateNodes(canvas: JsonCanvas, offsetX: number, offsetY: number): JsonCanvas {
    return {
      nodes: canvas.nodes.map((n) => ({ ...n, x: n.x + offsetX, y: n.y + offsetY })),
      edges: canvas.edges,
    }
  }

  /**
   * Scale node positions and sizes
   */
  static scaleCanvas(canvas: JsonCanvas, scale: number): JsonCanvas {
    return {
      nodes: canvas.nodes.map((n) => ({
        ...n,
        x: n.x * scale,
        y: n.y * scale,
        width: n.width * scale,
        height: n.height * scale,
      })),
      edges: canvas.edges,
    }
  }
}
