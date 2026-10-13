import type {
  QuartzEmitterPluginInstance,
  QuartzFilterPluginInstance,
  QuartzTransformerPluginInstance,
} from '../types/plugin'
import type { BuildCtx } from '../util/ctx'
import { isDevelopmentBuild } from '../util/build-mode'

type QuartzPluginInstance =
  | QuartzTransformerPluginInstance
  | QuartzFilterPluginInstance
  | QuartzEmitterPluginInstance

export type ConditionalPlugin<P extends QuartzPluginInstance> =
  | P
  | { plugin: P; dev: boolean; fallback?: NoInfer<P> }

function pluginKind(plugin: QuartzPluginInstance): string {
  if ('emit' in plugin) return 'emitter'
  if ('shouldPublish' in plugin) return 'filter'
  return 'transformer'
}

function isEmitter(plugin: QuartzPluginInstance): plugin is QuartzEmitterPluginInstance {
  return 'emit' in plugin && typeof plugin.emit === 'function'
}

function isFilter(plugin: QuartzPluginInstance): plugin is QuartzFilterPluginInstance {
  return 'shouldPublish' in plugin && typeof plugin.shouldPublish === 'function'
}

function wrapPlugin<P extends QuartzPluginInstance>(plugin: P, fallback?: P): P {
  if (fallback && pluginKind(plugin) !== pluginKind(fallback)) {
    throw new TypeError(`Fallback for ${plugin.name} must be a ${pluginKind(plugin)}`)
  }
  const enabled = (ctx: BuildCtx) => !isDevelopmentBuild(ctx.argv)
  if (isFilter(plugin)) {
    const alternate = fallback && isFilter(fallback) ? fallback : undefined
    return {
      ...plugin,
      shouldPublish: (...args: Parameters<typeof plugin.shouldPublish>) => {
        const active = enabled(args[0]) ? plugin : alternate
        return active?.shouldPublish(...args) ?? true
      },
    }
  }

  const resources = 'externalResources' in plugin ? plugin.externalResources : undefined
  const fallbackResources =
    fallback && 'externalResources' in fallback ? fallback.externalResources : undefined
  const externalResources =
    resources || fallbackResources
      ? (ctx: BuildCtx) =>
          enabled(ctx) ? resources?.call(plugin, ctx) : fallbackResources?.call(fallback, ctx)
      : undefined

  if (isEmitter(plugin)) {
    const alternate = fallback && isEmitter(fallback) ? fallback : undefined
    return {
      ...plugin,
      externalResources,
      emit: (...args: Parameters<typeof plugin.emit>) => {
        const active = enabled(args[0]) ? plugin : alternate
        return active?.emit(...args) ?? Promise.resolve([])
      },
      partialEmit:
        plugin.partialEmit || alternate?.partialEmit
          ? (...args: Parameters<NonNullable<QuartzEmitterPluginInstance['partialEmit']>>) => {
              const [ctx, content, resources, changes] = args
              const active = enabled(ctx) ? plugin : alternate
              if (!active) return null
              return active.partialEmit
                ? active.partialEmit(ctx, content, resources, changes)
                : active.emit(ctx, content, resources)
            }
          : undefined,
      getQuartzComponents:
        plugin.getQuartzComponents || alternate?.getQuartzComponents
          ? (ctx: BuildCtx) => {
              const active = enabled(ctx) ? plugin : alternate
              return active?.getQuartzComponents?.(ctx) ?? []
            }
          : undefined,
    }
  }

  const primary: QuartzTransformerPluginInstance = plugin
  const alternate: QuartzTransformerPluginInstance | undefined = fallback
  const { textTransform, markdownPlugins, htmlPlugins } = primary
  return {
    ...plugin,
    externalResources,
    textTransform:
      textTransform || alternate?.textTransform
        ? (ctx: BuildCtx, source: string) => {
            const active = enabled(ctx) ? primary : alternate
            return active?.textTransform?.(ctx, source) ?? source
          }
        : undefined,
    markdownPlugins:
      markdownPlugins || alternate?.markdownPlugins
        ? (ctx: BuildCtx) => {
            const active = enabled(ctx) ? primary : alternate
            return active?.markdownPlugins?.(ctx) ?? []
          }
        : undefined,
    htmlPlugins:
      htmlPlugins || alternate?.htmlPlugins
        ? (ctx: BuildCtx) => {
            const active = enabled(ctx) ? primary : alternate
            return active?.htmlPlugins?.(ctx) ?? []
          }
        : undefined,
  }
}

/** Keep array order and names; dev: false selects the fallback in unforced watch builds. */
export function Conditional<P extends QuartzPluginInstance>({
  plugins,
}: {
  plugins: readonly ConditionalPlugin<P>[]
}): P[] {
  return plugins.map(entry => {
    if (!('plugin' in entry)) return entry
    return entry.dev ? entry.plugin : wrapPlugin(entry.plugin, entry.fallback)
  })
}

export { Conditional as ConditionalTransform }
