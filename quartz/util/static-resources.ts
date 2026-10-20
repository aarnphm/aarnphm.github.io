import type { ProcessedContent } from '../plugins/vfile'
import type { BuildCtx } from './ctx'
import type { StaticResources } from './resources'

export function getStaticResourcesFromPlugins(ctx: BuildCtx, content?: ProcessedContent[]) {
  const staticResources: StaticResources = { css: [], js: [], additionalHead: [] }

  for (const transformer of [...ctx.cfg.plugins.transformers, ...ctx.cfg.plugins.emitters]) {
    const res = transformer.externalResources ? transformer.externalResources(ctx, content) : {}
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

  if (ctx.argv.serve || ctx.argv.watch) {
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
        // Any HTTP response means the server is up; a 404 page must still reload.
        // Only a rejected fetch (server down) retries, and a dead server stops after 2 minutes.
        const waitForQuartzReloadTarget = async () => {
          const deadline = Date.now() + 120_000
          while (Date.now() < deadline) {
            try {
              await fetch(window.location.href, { cache: "no-store" })
              return true
            } catch {}
            await quartzReloadDelay(250)
          }
          return false
        }
        socket.addEventListener('message', () => {
          if (quartzReloading) return
          quartzReloading = true
          waitForQuartzReloadTarget().then(ready => {
            if (ready) window.location.reload()
            else quartzReloading = false
          })
        })
      `,
    })
  }

  return staticResources
}
