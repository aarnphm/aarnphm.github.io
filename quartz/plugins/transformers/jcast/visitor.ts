import { visit as unistVisit, SKIP, CONTINUE, EXIT } from "unist-util-visit"
import {
  JcastNode,
  JcastCanvas,
  JcastCanvasNode,
  JcastCanvasEdge,
  JcastCanvasGroup,
  JcastVisitor,
  JcastGraphVisitor,
} from "./types"

/**
 * Visit nodes in a jcast tree (unist-compatible)
 *
 * This is a thin wrapper around unist-util-visit that provides
 * jcast-specific type safety.
 */
export function visitJcast<T extends JcastNode = JcastNode>(
  tree: JcastNode,
  type: string | string[],
  visitor: JcastVisitor<T>,
): void {
  unistVisit(tree, type, visitor as any)
}

/**
 * Visit all nodes in a jcast tree
 */
export function visitAllJcast(tree: JcastNode, visitor: JcastVisitor): void {
  unistVisit(tree, visitor as any)
}

/**
 * Visit nodes with edge context (graph-aware traversal)
 *
 * This visitor provides additional context about incoming and outgoing edges
 * for each node, making it easier to implement graph algorithms.
 */
export function visitGraph(canvas: JcastCanvas, visitor: JcastGraphVisitor): void {
  visitJcast(canvas, ["canvasNode", "canvasGroup"], (node, index, parent) => {
    const canvasNode = node as JcastCanvasNode | JcastCanvasGroup

    // gather incoming and outgoing edges
    const incomingEdges: JcastCanvasEdge[] = []
    const outgoingEdges: JcastCanvasEdge[] = []

    if (canvasNode.data?.incomingEdges) {
      for (const edgeId of canvasNode.data.incomingEdges) {
        const edge = canvas.data.edgeMap.get(edgeId)
        if (edge) incomingEdges.push(edge)
      }
    }

    if (canvasNode.data?.outgoingEdges) {
      for (const edgeId of canvasNode.data.outgoingEdges) {
        const edge = canvas.data.edgeMap.get(edgeId)
        if (edge) outgoingEdges.push(edge)
      }
    }

    return visitor(canvasNode, { incoming: incomingEdges, outgoing: outgoingEdges }, index, parent)
  })
}

/**
 * Walk connected nodes starting from a given node
 *
 * Performs breadth-first traversal of the graph following edges.
 */
export function walkConnected(
  canvas: JcastCanvas,
  startNodeId: string,
  visitor: JcastGraphVisitor,
  options: { direction?: "incoming" | "outgoing" | "both"; maxDepth?: number } = {},
): void {
  const { direction = "both", maxDepth = Infinity } = options

  const visited = new Set<string>()
  const queue: Array<{ nodeId: string; depth: number }> = [{ nodeId: startNodeId, depth: 0 }]

  while (queue.length > 0) {
    const { nodeId, depth } = queue.shift()!

    if (visited.has(nodeId) || depth > maxDepth) continue
    visited.add(nodeId)

    const node = canvas.data.nodeMap.get(nodeId)
    if (!node) continue

    // gather edges
    const incomingEdges: JcastCanvasEdge[] = []
    const outgoingEdges: JcastCanvasEdge[] = []

    if (node.data?.incomingEdges) {
      for (const edgeId of node.data.incomingEdges) {
        const edge = canvas.data.edgeMap.get(edgeId)
        if (edge) incomingEdges.push(edge)
      }
    }

    if (node.data?.outgoingEdges) {
      for (const edgeId of node.data.outgoingEdges) {
        const edge = canvas.data.edgeMap.get(edgeId)
        if (edge) outgoingEdges.push(edge)
      }
    }

    // call visitor
    const result = visitor(node, { incoming: incomingEdges, outgoing: outgoingEdges })
    if (result === false) continue
    if (result === true) break

    // add connected nodes to queue
    if (direction === "outgoing" || direction === "both") {
      for (const edge of outgoingEdges) {
        if (!visited.has(edge.data.toNode)) {
          queue.push({ nodeId: edge.data.toNode, depth: depth + 1 })
        }
      }
    }

    if (direction === "incoming" || direction === "both") {
      for (const edge of incomingEdges) {
        if (!visited.has(edge.data.fromNode)) {
          queue.push({ nodeId: edge.data.fromNode, depth: depth + 1 })
        }
      }
    }
  }
}

/**
 * Find a path between two nodes using BFS
 *
 * Returns an array of node IDs representing the path, or null if no path exists.
 */
export function findPath(
  canvas: JcastCanvas,
  startNodeId: string,
  endNodeId: string,
  options: { direction?: "incoming" | "outgoing" | "both" } = {},
): string[] | null {
  const { direction = "both" } = options

  const visited = new Set<string>()
  const parent = new Map<string, string>()
  const queue: string[] = [startNodeId]

  visited.add(startNodeId)

  while (queue.length > 0) {
    const nodeId = queue.shift()!

    if (nodeId === endNodeId) {
      // reconstruct path
      const path: string[] = []
      let current = endNodeId
      while (current !== startNodeId) {
        path.unshift(current)
        current = parent.get(current)!
      }
      path.unshift(startNodeId)
      return path
    }

    const node = canvas.data.nodeMap.get(nodeId)
    if (!node) continue

    // explore neighbors
    const neighbors: string[] = []

    if (direction === "outgoing" || direction === "both") {
      if (node.data?.outgoingEdges) {
        for (const edgeId of node.data.outgoingEdges) {
          const edge = canvas.data.edgeMap.get(edgeId)
          if (edge) neighbors.push(edge.data.toNode)
        }
      }
    }

    if (direction === "incoming" || direction === "both") {
      if (node.data?.incomingEdges) {
        for (const edgeId of node.data.incomingEdges) {
          const edge = canvas.data.edgeMap.get(edgeId)
          if (edge) neighbors.push(edge.data.fromNode)
        }
      }
    }

    for (const neighborId of neighbors) {
      if (!visited.has(neighborId)) {
        visited.add(neighborId)
        parent.set(neighborId, nodeId)
        queue.push(neighborId)
      }
    }
  }

  return null // no path found
}

/**
 * Find all nodes reachable from a given node
 */
export function findReachableNodes(
  canvas: JcastCanvas,
  startNodeId: string,
  options: { direction?: "incoming" | "outgoing" | "both"; maxDepth?: number } = {},
): Set<string> {
  const reachable = new Set<string>()

  walkConnected(
    canvas,
    startNodeId,
    (node) => {
      reachable.add(node.id!)
    },
    options,
  )

  return reachable
}

/**
 * Find strongly connected components using Tarjan's algorithm
 */
export function findStronglyConnectedComponents(canvas: JcastCanvas): string[][] {
  const index = new Map<string, number>()
  const lowlink = new Map<string, number>()
  const onStack = new Set<string>()
  const stack: string[] = []
  const components: string[][] = []
  let currentIndex = 0

  function strongConnect(nodeId: string) {
    index.set(nodeId, currentIndex)
    lowlink.set(nodeId, currentIndex)
    currentIndex++
    stack.push(nodeId)
    onStack.add(nodeId)

    const node = canvas.data.nodeMap.get(nodeId)
    if (node?.data?.outgoingEdges) {
      for (const edgeId of node.data.outgoingEdges) {
        const edge = canvas.data.edgeMap.get(edgeId)
        if (!edge) continue

        const targetId = edge.data.toNode
        if (!index.has(targetId)) {
          strongConnect(targetId)
          lowlink.set(nodeId, Math.min(lowlink.get(nodeId)!, lowlink.get(targetId)!))
        } else if (onStack.has(targetId)) {
          lowlink.set(nodeId, Math.min(lowlink.get(nodeId)!, index.get(targetId)!))
        }
      }
    }

    if (lowlink.get(nodeId) === index.get(nodeId)) {
      const component: string[] = []
      let w: string
      do {
        w = stack.pop()!
        onStack.delete(w)
        component.push(w)
      } while (w !== nodeId)
      components.push(component)
    }
  }

  for (const nodeId of canvas.data.nodeMap.keys()) {
    if (!index.has(nodeId)) {
      strongConnect(nodeId)
    }
  }

  return components
}

/**
 * Topological sort of nodes (returns null if graph has cycles)
 */
export function topologicalSort(canvas: JcastCanvas): string[] | null {
  const inDegree = new Map<string, number>()
  const result: string[] = []
  const queue: string[] = []

  // calculate in-degrees
  for (const nodeId of canvas.data.nodeMap.keys()) {
    inDegree.set(nodeId, 0)
  }

  for (const edge of canvas.data.edgeMap.values()) {
    const degree = inDegree.get(edge.data.toNode) || 0
    inDegree.set(edge.data.toNode, degree + 1)
  }

  // find nodes with in-degree 0
  for (const [nodeId, degree] of inDegree) {
    if (degree === 0) {
      queue.push(nodeId)
    }
  }

  while (queue.length > 0) {
    const nodeId = queue.shift()!
    result.push(nodeId)

    const node = canvas.data.nodeMap.get(nodeId)
    if (node?.data?.outgoingEdges) {
      for (const edgeId of node.data.outgoingEdges) {
        const edge = canvas.data.edgeMap.get(edgeId)
        if (!edge) continue

        const targetId = edge.data.toNode
        const degree = inDegree.get(targetId)! - 1
        inDegree.set(targetId, degree)

        if (degree === 0) {
          queue.push(targetId)
        }
      }
    }
  }

  // if result doesn't contain all nodes, there's a cycle
  if (result.length !== canvas.data.nodeMap.size) {
    return null
  }

  return result
}

// re-export unist-util-visit constants
export { SKIP, CONTINUE, EXIT }
