/**
 * Projections for the 3-d bench of `layout: latent`, top-k queries, and the steering channels.
 * The math lives in util/latent/vectors.ts, which the build also runs for latent embeds, so an
 * embed's build-time results are the explorer's.
 */
import { VectorEngine, type LatentWorkerRequest } from '../util/latent/vectors'

export type { LatentWorkerRequest, LatentWorkerResponse } from '../util/latent/vectors'

const engine = new VectorEngine(async url => {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${response.status} ${url}`)
  return response.arrayBuffer()
})

self.onmessage = async (event: MessageEvent<LatentWorkerRequest>) => {
  const { response, transfer } = await engine.handle(event.data)
  ;(self as unknown as Worker).postMessage(response, transfer)
}
