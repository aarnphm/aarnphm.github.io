import { build as bundle } from 'esbuild'
import fs from 'node:fs/promises'
import path from 'path'
import type { QuartzMdxComponent } from '../../../components/mdx/registry'
import type { QuartzComponent } from '../../../types/component'
import type { BuildCtx } from '../../../util/ctx'
import { getMdxComponents } from '../../../components/mdx/registry'
import notFoundScript from '../../../components/scripts/404.inline'
import audioScript from '../../../components/scripts/audio.inline'
import baseMapScript from '../../../components/scripts/base-map.inline'
import pseudoScript from '../../../components/scripts/clipboard-pseudo.inline'
import clipboardScript from '../../../components/scripts/clipboard.inline'
import collaborativeCommentsScript from '../../../components/scripts/collaborative-comments.inline'
import markerScript from '../../../components/scripts/marker.inline'
import petScript from '../../../components/scripts/pet.inline'
import popoverScript from '../../../components/scripts/popover.inline'
import protectedScript from '../../../components/scripts/protected.inline'
import spaRouterScript from '../../../components/scripts/spa.inline'
import audioStyle from '../../../components/styles/audio.scss'
import clipboardStyle from '../../../components/styles/clipboard.scss'
import '../../../components/mdx'
import popoverStyle from '../../../components/styles/popover.scss'
import pseudoStyle from '../../../components/styles/pseudocode.scss'
import { notebookRuntimeInlineEntry } from './asset-paths'

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

async function bundleInlineScript(scriptPath: string): Promise<string> {
  let text = await fs.readFile(scriptPath, 'utf8')
  text = text.replace('export default', '')
  text = text.replace('export', '')

  const sourcefile = path.relative(path.resolve('.'), scriptPath)
  const transpiled = await bundle({
    stdin: {
      contents: text,
      loader: path.extname(scriptPath) === '.js' ? 'js' : 'ts',
      resolveDir: path.dirname(sourcefile),
      sourcefile,
    },
    write: false,
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    loader: { '.py': 'text' },
  })
  return transpiled.outputFiles[0].text
}

function addGlobalPageResources(ctx: BuildCtx, componentResources: ComponentResourceSet) {
  const cfg = ctx.cfg.configuration

  if (cfg.enablePopovers) {
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
  addGlobalPageResources(ctx, componentResources)
  return componentResources
}
