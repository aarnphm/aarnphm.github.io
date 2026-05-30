import type { BuildCtx } from './ctx'
import type { StaticResources } from './resources'

export function getStaticResourcesFromPlugins(ctx: BuildCtx) {
  const staticResources: StaticResources = { css: [], js: [], additionalHead: [] }

  for (const transformer of [...ctx.cfg.plugins.transformers, ...ctx.cfg.plugins.emitters]) {
    const res = transformer.externalResources ? transformer.externalResources(ctx) : {}
    if (res?.js) {
      staticResources.js.push(...res.js)
    }
    if (res?.css) {
      staticResources.css.push(...res.css)
    }
    if (res?.additionalHead) {
      staticResources.additionalHead.push(...res.additionalHead)
    }
  }

  if (ctx.argv.serve) {
    const wsUrl = ctx.argv.remoteDevHost
      ? `wss://${ctx.argv.remoteDevHost}:${ctx.argv.wsPort}`
      : `ws://localhost:${ctx.argv.wsPort}`

    staticResources.js.push({
      loadTime: 'afterDOMReady',
      contentType: 'inline',
      script: `
        const socket = new WebSocket('${wsUrl}')
        socket.addEventListener('message', () => document.location.reload(true))
      `,
    })
  }

  return staticResources
}
