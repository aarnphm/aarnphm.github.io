import type { QuartzMdxComponent } from '../../../components/mdx/registry'
import type { QuartzComponent } from '../../../types/component'
import type { BuildCtx } from '../../../util/ctx'
import { getMdxComponents } from '../../../components/mdx/registry'
import audioStyle from '../../../components/styles/audio.scss'
import clipboardStyle from '../../../components/styles/clipboard.scss'
import popoverStyle from '../../../components/styles/popover.scss'
import '../../../components/mdx'
import pseudoStyle from '../../../components/styles/pseudocode.scss'
import { bundleInlineScript } from '../../../util/inline-script-bundler'
import { notebookRuntimeInlineEntry } from './asset-paths'

const notFoundInlineEntry = 'quartz/components/scripts/404.inline.ts'
const audioInlineEntry = 'quartz/components/scripts/audio.inline.ts'
const baseMapInlineEntry = 'quartz/components/scripts/base-map.inline.ts'
const pseudoInlineEntry = 'quartz/components/scripts/clipboard-pseudo.inline.ts'
const clipboardInlineEntry = 'quartz/components/scripts/clipboard.inline.ts'
const collaborativeCommentsInlineEntry =
  'quartz/components/scripts/collaborative-comments.inline.ts'
const markerInlineEntry = 'quartz/components/scripts/marker.inline.ts'
const petInlineEntry = 'quartz/components/scripts/pet.inline.ts'
const popoverInlineEntry = 'quartz/components/scripts/popover.inline.ts'
const protectedInlineEntry = 'quartz/components/scripts/protected.inline.ts'
const spaInlineEntry = 'quartz/components/scripts/spa.inline.ts'

export type ComponentResourceSet = {
  css: string[]
  componentCss: string[]
  beforeDOMLoaded: string[]
  afterDOMLoaded: string[]
}

export function normalizeResource(resource: string | string[] | undefined): string[] {
  if (!resource) return []
  if (Array.isArray(resource)) return resource
  return [resource]
}

function getComponentResources(ctx: BuildCtx): ComponentResourceSet {
  const allComponents: Set<QuartzComponent | QuartzMdxComponent> = new Set()
  for (const emitter of ctx.cfg.plugins.emitters) {
    const components = emitter.getQuartzComponents?.(ctx) ?? []
    for (const component of components) {
      allComponents.add(component)
    }
  }
  for (const component of getMdxComponents()) {
    allComponents.add(component)
  }

  const componentResources = {
    css: new Set<string>(),
    beforeDOMLoaded: new Set<string>(),
    afterDOMLoaded: new Set<string>(),
  }

  for (const component of allComponents) {
    const { css, beforeDOMLoaded, afterDOMLoaded } = component
    normalizeResource(css).forEach(c => componentResources.css.add(c))
    normalizeResource(beforeDOMLoaded).forEach(b => componentResources.beforeDOMLoaded.add(b))
    normalizeResource(afterDOMLoaded).forEach(a => componentResources.afterDOMLoaded.add(a))
  }

  return {
    css: [...componentResources.css],
    componentCss: [...componentResources.css],
    beforeDOMLoaded: [...componentResources.beforeDOMLoaded],
    afterDOMLoaded: [...componentResources.afterDOMLoaded],
  }
}

function notebookRuntimeInlineResourceIndex(componentResources: ComponentResourceSet): number {
  return componentResources.afterDOMLoaded.findIndex(
    script =>
      script.includes('notebookRuntimeScriptUrl') && script.includes('data-notebook-runtime-data'),
  )
}

async function refreshNotebookRuntimeInlineResource(componentResources: ComponentResourceSet) {
  const index = notebookRuntimeInlineResourceIndex(componentResources)
  if (index < 0) return
  componentResources.afterDOMLoaded[index] = await bundleInlineScript(notebookRuntimeInlineEntry)
}

async function addGlobalPageResources(ctx: BuildCtx, componentResources: ComponentResourceSet) {
  const cfg = ctx.cfg.configuration
  const [
    notFoundScript,
    audioScript,
    baseMapScript,
    pseudoScript,
    clipboardScript,
    collaborativeCommentsScript,
    markerScript,
    petScript,
    protectedScript,
    spaRouterScript,
  ] = await Promise.all([
    bundleInlineScript(notFoundInlineEntry),
    bundleInlineScript(audioInlineEntry),
    bundleInlineScript(baseMapInlineEntry),
    bundleInlineScript(pseudoInlineEntry),
    bundleInlineScript(clipboardInlineEntry),
    bundleInlineScript(collaborativeCommentsInlineEntry),
    bundleInlineScript(markerInlineEntry),
    bundleInlineScript(petInlineEntry),
    bundleInlineScript(protectedInlineEntry),
    bundleInlineScript(spaInlineEntry),
  ])

  if (cfg.enablePopovers) {
    const popoverScript = await bundleInlineScript(popoverInlineEntry)
    componentResources.afterDOMLoaded.push(popoverScript)
    componentResources.css.push(popoverStyle)
  }

  componentResources.beforeDOMLoaded.push(markerScript)
  componentResources.css.push(clipboardStyle, pseudoStyle, audioStyle)
  componentResources.afterDOMLoaded.push(
    clipboardScript,
    pseudoScript,
    protectedScript,
    audioScript,
    baseMapScript,
    collaborativeCommentsScript,
    petScript,
  )

  if (cfg.analytics?.provider === 'plausible') {
    const plausibleHost = cfg.analytics.host ?? 'https://plausible.io'
    componentResources.afterDOMLoaded.push(`
      const plausibleScript = document.createElement("script")
      plausibleScript.src = "${plausibleHost}/js/script.outbound-links.manual.js"
      plausibleScript.setAttribute("data-domain", [location.hostname, "stream.aarnphm.xyz"].join(','))
      plausibleScript.setAttribute("data-api", "/_plausible/event")
      plausibleScript.dataset.persist = "true"
      plausibleScript.defer = true
      plausibleScript.onload = () => {
        window.plausible = window.plausible || function () { (window.plausible.q = window.plausible.q || []).push(arguments); };
        plausible('pageview')
        document.addEventListener('nav', () => {
          plausible('pageview')
        })
      }

      document.head.appendChild(plausibleScript)
    `)
  }

  componentResources.afterDOMLoaded.push(notFoundScript, spaRouterScript)
}

export async function currentComponentResources(ctx: BuildCtx): Promise<ComponentResourceSet> {
  const componentResources = getComponentResources(ctx)
  await refreshNotebookRuntimeInlineResource(componentResources)
  await addGlobalPageResources(ctx, componentResources)
  return componentResources
}
