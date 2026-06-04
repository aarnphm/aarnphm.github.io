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
        let quartzReloading = false
        const quartzReloadDelay = ms => new Promise(resolve => setTimeout(resolve, ms))
        const waitForQuartzReloadTarget = async () => {
          for (;;) {
            try {
              const response = await fetch(window.location.href, { cache: "no-store" })
              if (response.ok) return
            } catch {}
            await quartzReloadDelay(250)
          }
        }
        socket.addEventListener('message', () => {
          if (quartzReloading) return
          quartzReloading = true
          waitForQuartzReloadTarget().then(() => window.location.reload())
        })
      `,
    })
  }

  return staticResources
}
