import type { PageStyle } from '../../util/page-styles'
import { pageScopedComponentStyles } from '../../plugins/emitters/component-resources/page-scoped-styles'
import { resolveAsset, resolveExtractedStaticResource } from '../../util/asset-manifest'
import { BuildCtx } from '../../util/ctx'
import { FullSlug, joinSegments, RelativeURL } from '../../util/path'
import {
  componentCssResourceKeyPrefix,
  componentCssScopeKey,
  splitCssBundles,
  splitJsBundles,
  staticCssBundleKey,
  staticJsBundleKey,
} from '../../util/resource-bundles'
import { CSSResource, JSResource, StaticResources } from '../../util/resources'
import collapseHeaderStyle from '../styles/collapseHeader.inline.scss'

export const pageResources = (
  baseDir: FullSlug | RelativeURL,
  staticResources: StaticResources,
  ctx: BuildCtx,
) => {
  if (ctx.pageResourceCacheBuildId !== ctx.buildId) {
    ctx.pageResourceCacheBuildId = ctx.buildId
    ctx.pageResourceCache = new Map()
  }
  const cacheKey = `${baseDir}`
  const cached = ctx.pageResourceCache?.get(cacheKey)
  if (cached) return cached

  const asset = (logicalPath: string) => joinSegments(baseDir, resolveAsset(ctx, logicalPath))
  const extractedAsset = (key: string) =>
    joinSegments(baseDir, resolveExtractedStaticResource(ctx, key))
  const componentCss: CSSResource[] = [
    ...(ctx.extractedStaticResources ?? new Map<string, string>()).entries(),
  ]
    .filter(([key]) => key.startsWith(componentCssResourceKeyPrefix))
    .map(([key, content]) => ({
      content: joinSegments(baseDir, content),
      classPrefixes: pageScopedComponentStyles.find(
        style => componentCssScopeKey(style.name) === key,
      )?.classPrefixes,
    }))
  const css: CSSResource[] = [
    { content: asset('index.css') },
    ...componentCss,
    ...splitCssBundles(staticResources.css, [collapseHeaderStyle]).map(part =>
      part.type === 'bundle'
        ? { content: extractedAsset(staticCssBundleKey(part.content)) }
        : part.resource,
    ),
  ]
  const pageStyles: PageStyle[] = css.flatMap(({ content, classPrefixes }) =>
    classPrefixes ? [{ href: content, classPrefixes }] : [],
  )
  const staticJs: JSResource[] = []
  for (const part of [
    ...splitJsBundles(
      staticResources.js,
      'beforeDOMReady',
      ctx.staticLeadingJs?.beforeDOMReady ?? [],
    ),
    ...splitJsBundles(
      staticResources.js,
      'afterDOMReady',
      ctx.staticLeadingJs?.afterDOMReady ?? [],
    ),
  ]) {
    if (part.type === 'bundle') {
      const resource: JSResource = {
        src: extractedAsset(staticJsBundleKey(part.loadTime, part.scripts)),
        loadTime: part.loadTime,
        contentType: 'external',
      }
      staticJs.push(resource)
    } else {
      staticJs.push(part.resource)
    }
  }

  const resources = {
    css: [...css],
    js: [
      { src: asset('prescript.js'), loadTime: 'beforeDOMReady', contentType: 'external' },
      {
        loadTime: 'beforeDOMReady',
        contentType: 'inline',
        spaPreserve: true,
        script: `const lazyFetchJson = (url, loader) => { let promise; const load = () => promise ??= (loader ?? (target => fetch(target).then(data => data.json())))(url); return { then: (onFulfilled, onRejected) => load().then(onFulfilled, onRejected), catch: onRejected => load().catch(onRejected), finally: onFinally => load().finally(onFinally), get [Symbol.toStringTag]() { return "Promise" } } }; const loadSearchIndex = async url => { const data = await fetch(url).then(resp => resp.json()); if (!data || data.kind !== "quartz-search-index-v1" || !Array.isArray(data.chunks)) return data; const base = new URL(url, window.location.href); const chunks = await Promise.all(data.chunks.map(chunk => fetch(new URL(chunk, base)).then(resp => resp.json()))); return Object.assign({}, ...chunks) }; const fetchData = lazyFetchJson("${joinSegments(baseDir, 'static/contentIndex.json')}")`,
      },
      {
        loadTime: 'beforeDOMReady',
        contentType: 'inline',
        spaPreserve: true,
        script: `const fetchSearchData = lazyFetchJson("${joinSegments(baseDir, 'static/searchIndex.json')}", loadSearchIndex)`,
      },
      {
        loadTime: 'beforeDOMReady',
        contentType: 'inline',
        spaPreserve: true,
        script: `const semanticCfg = ${JSON.stringify(ctx.cfg?.configuration?.semanticSearch ?? {})}`,
      },
      {
        loadTime: 'beforeDOMReady',
        contentType: 'inline',
        spaPreserve: true,
        // Absolute hrefs: SPA navigation changes the base that relative hrefs resolve against.
        script: `const pageStyles = ${JSON.stringify(pageStyles)}.map(style => ({ ...style, href: new URL(style.href, document.baseURI).href }))`,
      },
      ...staticJs,
      {
        src: asset('postscript.js'),
        loadTime: 'afterDOMReady',
        moduleType: 'module',
        contentType: 'external',
        crossOrigin: 'anonymous',
      },
    ],
    additionalHead: staticResources.additionalHead,
  } satisfies StaticResources

  ctx.pageResourceCache!.set(cacheKey, resources)
  return resources
}
