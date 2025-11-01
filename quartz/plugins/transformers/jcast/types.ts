import { Node as UnistNode, Data as UnistData, Parent } from "unist"

/**
 * jcast - JSON Canvas Abstract Syntax Tree
 *
 * A unist-compatible AST for JSON Canvas files that maintains both
 * tree structure (for traversal) and graph relationships (for canvas semantics).
 */

// ===== JSON Canvas Spec Types =====

export type NodeType = "text" | "file" | "link" | "group"
export type EdgeEnd = "none" | "arrow"
export type Side = "top" | "right" | "bottom" | "left"

/**
 * Raw JSON Canvas node from .canvas file
 */
export interface JsonCanvasNode {
  id: string
  type: NodeType
  x: number
  y: number
  width: number
  height: number
  color?: string
  // text node properties
  text?: string
  // file node properties
  file?: string
  subpath?: string
  // link node properties
  url?: string
  // group node properties
  label?: string
  background?: string
  backgroundStyle?: string
}

/**
 * Raw JSON Canvas edge from .canvas file
 */
export interface JsonCanvasEdge {
  id: string
  fromNode: string
  fromSide?: Side
  fromEnd?: EdgeEnd
  toNode: string
  toSide?: Side
  toEnd?: EdgeEnd
  color?: string
  label?: string
}

/**
 * Complete JSON Canvas file structure
 */
export interface JsonCanvas {
  nodes: JsonCanvasNode[]
  edges: JsonCanvasEdge[]
}

// ===== jcast Node Types =====

/**
 * Base jcast node data - extends unist Data
 */
export interface JcastData extends UnistData {
  /**
   * Original JSON Canvas properties preserved in data.canvas
   */
  canvas?: Partial<JsonCanvasNode> & Partial<JsonCanvasEdge>

  /**
   * Graph relationship metadata
   */
  edges?: string[] // IDs of connected edges
  incomingEdges?: string[] // IDs of edges pointing to this node
  outgoingEdges?: string[] // IDs of edges from this node
  containedNodes?: string[] // IDs of nodes within this group (for group nodes)
  parentGroup?: string // ID of parent group (if any)

  /**
   * Resolved content for file nodes
   */
  resolvedPath?: string // Resolved file path
  fileContent?: string // Markdown content (loaded lazily)
  fileExists?: boolean // Whether the file exists

  /**
   * Additional resolved metadata for navigation and UI
   */
  resolved?: {
    slug: string
    href: string
    displayName: string
    description?: string
    content?: string
  }

  /**
   * Layout and rendering metadata
   */
  computedPosition?: { x: number; y: number } // Position after layout algorithm
  renderOrder?: number // Z-index for rendering
}

/**
 * Base jcast node - compatible with unist Node
 */
export interface JcastNode extends UnistNode {
  type: "canvas" | "canvasNode" | "canvasEdge" | "canvasGroup"
  id?: string
  data?: JcastData
  children?: JcastNode[]
}

/**
 * Root canvas node - represents entire .canvas file
 */
export interface JcastCanvas extends JcastNode {
  type: "canvas"
  children: JcastNode[] // All nodes and edges
  data: JcastData & {
    /**
     * Fast lookup maps for graph operations
     */
    nodeMap: Map<string, JcastCanvasNode | JcastCanvasGroup>
    edgeMap: Map<string, JcastCanvasEdge>

    /**
     * Canvas-level metadata
     */
    bounds?: {
      minX: number
      minY: number
      maxX: number
      maxY: number
    }
  }
}

/**
 * Canvas node - represents a node in the canvas
 */
export interface JcastCanvasNode extends JcastNode {
  type: "canvasNode"
  id: string
  data: JcastData & {
    canvas: JsonCanvasNode
    nodeType: NodeType
  }
  children?: JcastNode[] // For groups, contains child nodes
}

/**
 * Canvas edge - represents a connection between nodes
 */
export interface JcastCanvasEdge extends JcastNode {
  type: "canvasEdge"
  id: string
  data: JcastData & {
    canvas: JsonCanvasEdge
    fromNode: string
    toNode: string
  }
}

/**
 * Canvas group - represents a group container
 */
export interface JcastCanvasGroup extends JcastNode {
  type: "canvasGroup"
  id: string
  data: JcastData & {
    canvas: JsonCanvasNode
    nodeType: "group"
  }
  children: JcastNode[] // Nodes contained in this group
}

// ===== Type Guards =====

export function isJcastCanvas(node: unknown): node is JcastCanvas {
  return typeof node === "object" && node !== null && (node as JcastNode).type === "canvas"
}

export function isJcastCanvasNode(node: unknown): node is JcastCanvasNode {
  return typeof node === "object" && node !== null && (node as JcastNode).type === "canvasNode"
}

export function isJcastCanvasEdge(node: unknown): node is JcastCanvasEdge {
  return typeof node === "object" && node !== null && (node as JcastNode).type === "canvasEdge"
}

export function isJcastCanvasGroup(node: unknown): node is JcastCanvasGroup {
  return typeof node === "object" && node !== null && (node as JcastNode).type === "canvasGroup"
}

// ===== Visitor Types =====

/**
 * Visitor function type for jcast traversal
 */
export type JcastVisitor<T extends JcastNode = JcastNode> = (
  node: T,
  index?: number,
  parent?: Parent,
) => void | boolean | Promise<void | boolean>

/**
 * Graph visitor function - receives node with edge context
 */
export type JcastGraphVisitor<T extends JcastNode = JcastNode> = (
  node: T,
  edges: {
    incoming: JcastCanvasEdge[]
    outgoing: JcastCanvasEdge[]
  },
  index?: number,
  parent?: Parent,
) => void | boolean | Promise<void | boolean>

// ===== Builder Types =====

/**
 * Builder options for creating canvas nodes
 */
export interface NodeBuilderOptions {
  id?: string
  x: number
  y: number
  width: number
  height: number
  color?: string
}

export interface TextNodeOptions extends NodeBuilderOptions {
  text: string
}

export interface FileNodeOptions extends NodeBuilderOptions {
  file: string
  subpath?: string
}

export interface LinkNodeOptions extends NodeBuilderOptions {
  url: string
}

export interface GroupNodeOptions extends NodeBuilderOptions {
  label?: string
  background?: string
  backgroundStyle?: string
}

/**
 * Builder options for creating edges
 */
export interface EdgeBuilderOptions {
  id?: string
  fromNode: string
  toNode: string
  fromSide?: Side
  fromEnd?: EdgeEnd
  toSide?: Side
  toEnd?: EdgeEnd
  color?: string
  label?: string
}

// ===== Layout Types =====

export type LayoutAlgorithm = "force-directed" | "hierarchical" | "radial" | "manual"

export interface LayoutOptions {
  algorithm: LayoutAlgorithm
  width?: number
  height?: number
  padding?: number
  // force-directed options
  forceStrength?: number
  linkDistance?: number
  // hierarchical options
  levelSeparation?: number
  nodeSeparation?: number
  direction?: "TB" | "BT" | "LR" | "RL"
  // radial options
  radius?: number
  startAngle?: number
}

/**
 * Position update callback for layout algorithms
 */
export type PositionUpdateCallback = (nodeId: string, x: number, y: number) => void
