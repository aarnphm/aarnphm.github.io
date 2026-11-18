import {
  forceSimulation,
  forceManyBody,
  forceCenter,
  forceLink,
  forceCollide,
  forceRadial,
  type Simulation,
} from 'd3'
import type { D3Config } from '../../Graph'
import type { NodeData, LinkData } from './data'

type SimulationOptions = Pick<
  D3Config,
  'repelForce' | 'centerForce' | 'linkDistance' | 'enableRadial'
>

export function createGraphSimulation(
  data: { nodes: NodeData[]; links: LinkData[] },
  viewport: { width: number; height: number },
  options: SimulationOptions,
): Simulation<NodeData, LinkData> {
  const nodeRadius = (node: NodeData) => node.radius
  const simulation: Simulation<NodeData, LinkData> = forceSimulation<NodeData>(data.nodes)
    .force('charge', forceManyBody().strength(-100 * options.repelForce))
    .force('center', forceCenter().strength(options.centerForce))
    .force('link', forceLink(data.links).distance(options.linkDistance))
    .force('collide', forceCollide<NodeData>(node => nodeRadius(node)).iterations(3))
  if (options.enableRadial)
    simulation.force(
      'radial',
      forceRadial((Math.min(viewport.width, viewport.height) / 2) * 0.8).strength(0.2),
    )
  return simulation
}
