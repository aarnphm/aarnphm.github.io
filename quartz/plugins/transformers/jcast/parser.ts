import {
  JsonCanvas,
  JsonCanvasNode,
  JsonCanvasEdge,
  JcastCanvas,
  JcastCanvasNode,
  JcastCanvasEdge,
  JcastCanvasGroup,
  JcastNode,
} from "./types"

/**
 * Parse a JSON Canvas file into a jcast AST
 *
 * This function transforms the raw JSON Canvas format into a unist-compatible
 * AST while preserving graph relationships through ID references and lookup maps.
 */
export function parseJsonCanvas(json: JsonCanvas | string): JcastCanvas {
  const canvas: JsonCanvas = typeof json === "string" ? JSON.parse(json) : json

  // initialize root canvas node
  const ast: JcastCanvas = {
    type: "canvas",
    children: [],
    data: { nodeMap: new Map(), edgeMap: new Map() },
  }

  // phase 1: transform all nodes
  const nodeMap = new Map<string, JcastCanvasNode | JcastCanvasGroup>()

  for (const node of canvas.nodes || []) {
    const jcastNode = transformNode(node)
    nodeMap.set(node.id, jcastNode)
  }

  // phase 2: transform all edges and build relationship metadata
  const edgeMap = new Map<string, JcastCanvasEdge>()

  for (const edge of canvas.edges || []) {
    const jcastEdge = transformEdge(edge)
    edgeMap.set(edge.id, jcastEdge)

    // update node metadata with edge references
    const fromNode = nodeMap.get(edge.fromNode)
    const toNode = nodeMap.get(edge.toNode)

    if (fromNode?.data) {
      if (!fromNode.data.edges) fromNode.data.edges = []
      if (!fromNode.data.outgoingEdges) fromNode.data.outgoingEdges = []
      fromNode.data.edges.push(edge.id)
      fromNode.data.outgoingEdges.push(edge.id)
    }

    if (toNode?.data) {
      if (!toNode.data.edges) toNode.data.edges = []
      if (!toNode.data.incomingEdges) toNode.data.incomingEdges = []
      toNode.data.edges.push(edge.id)
      toNode.data.incomingEdges.push(edge.id)
    }
  }

  // phase 3: build group hierarchies
  const groupMap = new Map<string, JcastCanvasGroup>()
  const ungroupedNodes: JcastNode[] = []

  for (const [id, node] of nodeMap) {
    if (node.type === "canvasGroup") {
      groupMap.set(id, node as JcastCanvasGroup)
      if (!node.children) node.children = []
    } else {
      ungroupedNodes.push(node)
    }
  }

  // assign nodes to groups based on spatial containment
  for (const node of ungroupedNodes) {
    const parentGroup = findContainingGroup(node as JcastCanvasNode, groupMap)
    if (parentGroup && node.data) {
      parentGroup.children!.push(node)
      node.data.parentGroup = parentGroup.id
      if (!parentGroup.data.containedNodes) parentGroup.data.containedNodes = []
      parentGroup.data.containedNodes.push(node.id!)
    } else {
      ast.children.push(node)
    }
  }

  // add groups to children
  for (const group of groupMap.values()) {
    ast.children.push(group)
  }

  // phase 4: auto-resize groups to fit children
  for (const group of groupMap.values()) {
    resizeGroupToFitChildren(group, 40)
  }

  // add edges to children
  for (const edge of edgeMap.values()) {
    ast.children.push(edge)
  }

  // store maps in ast data
  ast.data.nodeMap = nodeMap
  ast.data.edgeMap = edgeMap

  // compute canvas bounds
  ast.data.bounds = computeBounds(canvas.nodes || [])

  return ast
}

/**
 * Transform a JSON Canvas node to jcast node
 */
function transformNode(node: JsonCanvasNode): JcastCanvasNode | JcastCanvasGroup {
  if (node.type === "group") {
    return {
      type: "canvasGroup",
      id: node.id,
      data: { canvas: node, nodeType: node.type },
      children: [],
    }
  }

  return { type: "canvasNode", id: node.id, data: { canvas: node, nodeType: node.type } }
}

/**
 * Transform a JSON Canvas edge to jcast edge
 */
function transformEdge(edge: JsonCanvasEdge): JcastCanvasEdge {
  return {
    type: "canvasEdge",
    id: edge.id,
    data: { canvas: edge, fromNode: edge.fromNode, toNode: edge.toNode },
  }
}

/**
 * Find the group that spatially contains a node
 */
function findContainingGroup(
  node: JcastCanvasNode,
  groups: Map<string, JcastCanvasGroup>,
): JcastCanvasGroup | null {
  const nodeCanvas = node.data.canvas
  if (!nodeCanvas) return null

  // find all groups that contain this node spatially
  const containingGroups: JcastCanvasGroup[] = []

  for (const group of groups.values()) {
    const groupCanvas = group.data.canvas
    if (!groupCanvas) continue

    if (
      nodeCanvas.x >= groupCanvas.x &&
      nodeCanvas.y >= groupCanvas.y &&
      nodeCanvas.x + nodeCanvas.width <= groupCanvas.x + groupCanvas.width &&
      nodeCanvas.y + nodeCanvas.height <= groupCanvas.y + groupCanvas.height
    ) {
      containingGroups.push(group)
    }
  }

  // if multiple groups contain this node, return the smallest one (most specific)
  if (containingGroups.length === 0) return null
  if (containingGroups.length === 1) return containingGroups[0]

  return containingGroups.reduce((smallest, current) => {
    const smallestArea = smallest.data.canvas!.width * smallest.data.canvas!.height
    const currentArea = current.data.canvas!.width * current.data.canvas!.height
    return currentArea < smallestArea ? current : smallest
  })
}

/**
 * Compute bounding box for all nodes
 */
function computeBounds(nodes: JsonCanvasNode[]) {
  if (nodes.length === 0) {
    return { minX: 0, minY: 0, maxX: 0, maxY: 0 }
  }

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const node of nodes) {
    minX = Math.min(minX, node.x)
    minY = Math.min(minY, node.y)
    maxX = Math.max(maxX, node.x + node.width)
    maxY = Math.max(maxY, node.y + node.height)
  }

  return { minX, minY, maxX, maxY }
}

/**
 * Auto-resize a group node to fit all its children with padding
 */
function resizeGroupToFitChildren(group: JcastCanvasGroup, padding: number = 40): void {
  if (!group.children || group.children.length === 0) return

  const childNodes = group.children.filter(
    (node) => node.type === "canvasNode" || node.type === "canvasGroup",
  )

  if (childNodes.length === 0) return

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const child of childNodes) {
    const childCanvas = (child as JcastCanvasNode | JcastCanvasGroup).data?.canvas
    if (!childCanvas) continue

    minX = Math.min(minX, childCanvas.x)
    minY = Math.min(minY, childCanvas.y)
    maxX = Math.max(maxX, childCanvas.x + childCanvas.width)
    maxY = Math.max(maxY, childCanvas.y + childCanvas.height)
  }

  if (isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
    group.data.canvas!.x = minX - padding
    group.data.canvas!.y = minY - padding
    group.data.canvas!.width = maxX - minX + padding * 2
    group.data.canvas!.height = maxY - minY + padding * 2
  }
}

/**
 * Serialize jcast back to JSON Canvas format
 */
export function serializeJcast(ast: JcastCanvas): JsonCanvas {
  const nodes: JsonCanvasNode[] = []
  const edges: JsonCanvasEdge[] = []

  // extract nodes from AST
  for (const [, node] of ast.data.nodeMap) {
    if (node.data.canvas) {
      nodes.push(node.data.canvas as JsonCanvasNode)
    }
  }

  // extract edges from AST
  for (const [, edge] of ast.data.edgeMap) {
    if (edge.data.canvas) {
      edges.push(edge.data.canvas as JsonCanvasEdge)
    }
  }

  return { nodes, edges }
}
