import { ProcessedContent, QuartzPluginData } from '../../plugins/vfile'
import { ChangeEvent } from '../../types/plugin'
import { FilePath, FullSlug, simplifySlug, slugifyFilePath } from '../path'
import { renderBaseViewsForFile } from './render'

type RenderedBaseFile = ReturnType<typeof renderBaseViewsForFile>

export type BaseRenderPlan = {
  baseData: QuartzPluginData
  rendered: RenderedBaseFile
  memberSlugs: Set<string>
}

export type BaseViewPartialState = {
  baseMemberSlugs: Map<FullSlug, Set<string>>
  linksBySlug: Map<string, Set<string>>
  incomingLinksBySlug: Map<string, Set<string>>
  slugByPath: Map<FilePath, string>
}

export type BaseViewPartialPlan = {
  allFiles: QuartzPluginData[]
  basePlans: Map<FullSlug, BaseRenderPlan>
  nextState: BaseViewPartialState
  slugsToRebuild: Set<FullSlug>
}

const baseSlugForData = (data: QuartzPluginData): FullSlug | undefined =>
  data.bases && data.basesConfig && data.slug ? (data.slug as FullSlug) : undefined

const slugKeyForData = (data: QuartzPluginData): string | undefined =>
  data.slug ? simplifySlug(data.slug as FullSlug) : undefined

const slugKeyForPath = (path: FilePath): string => simplifySlug(slugifyFilePath(path))

const linksForData = (data: QuartzPluginData): Set<string> => {
  const links = Array.isArray(data.links) ? data.links : []
  return new Set(links.map(link => String(link)).filter(link => link.length > 0))
}

const sameSet = (left: Set<string> | undefined, right: Set<string> | undefined): boolean => {
  const leftSet = left ?? new Set<string>()
  const rightSet = right ?? new Set<string>()
  if (leftSet.size !== rightSet.size) return false
  for (const value of leftSet) {
    if (!rightSet.has(value)) return false
  }
  return true
}

const collectBaseFiles = (content: ProcessedContent[]): Map<FullSlug, QuartzPluginData> => {
  const baseFilesBySlug = new Map<FullSlug, QuartzPluginData>()
  for (const [, file] of content) {
    const slug = baseSlugForData(file.data)
    if (slug) {
      baseFilesBySlug.set(slug, file.data)
    }
  }
  return baseFilesBySlug
}

const collectBaseRenderPlans = (
  baseFilesBySlug: Map<FullSlug, QuartzPluginData>,
  allFiles: QuartzPluginData[],
): Map<FullSlug, BaseRenderPlan> => {
  const plans = new Map<FullSlug, BaseRenderPlan>()
  for (const [baseSlug, baseData] of baseFilesBySlug) {
    const rendered = renderBaseViewsForFile(baseData, allFiles, baseData)
    const memberSlugs = new Set<string>()
    for (const view of rendered.views) {
      for (const slug of view.matchedSlugs) {
        memberSlugs.add(slug)
      }
    }
    plans.set(baseSlug, { baseData, rendered, memberSlugs })
  }
  return plans
}

const collectBaseViewPartialState = (
  allFiles: QuartzPluginData[],
  basePlans: Map<FullSlug, BaseRenderPlan>,
): BaseViewPartialState => {
  const baseMemberSlugs = new Map<FullSlug, Set<string>>()
  for (const [baseSlug, plan] of basePlans) {
    baseMemberSlugs.set(baseSlug, new Set(plan.memberSlugs))
  }

  const linksBySlug = new Map<string, Set<string>>()
  const incomingLinksBySlug = new Map<string, Set<string>>()
  const slugByPath = new Map<FilePath, string>()
  for (const file of allFiles) {
    const slug = slugKeyForData(file)
    if (!slug) continue
    const links = linksForData(file)
    linksBySlug.set(slug, links)
    for (const link of links) {
      let sourceSlugs = incomingLinksBySlug.get(link)
      if (!sourceSlugs) {
        sourceSlugs = new Set()
        incomingLinksBySlug.set(link, sourceSlugs)
      }
      sourceSlugs.add(slug)
    }
    if (typeof file.relativePath === 'string') {
      slugByPath.set(file.relativePath as FilePath, slug)
    }
  }

  return { baseMemberSlugs, linksBySlug, incomingLinksBySlug, slugByPath }
}

const addBasesContainingSlug = (
  targetSlug: string,
  slugsToRebuild: Set<FullSlug>,
  currentPlans: Map<FullSlug, BaseRenderPlan>,
  previousState: BaseViewPartialState | undefined,
) => {
  for (const [baseSlug, plan] of currentPlans) {
    if (plan.memberSlugs.has(targetSlug)) {
      slugsToRebuild.add(baseSlug)
    }
  }

  if (!previousState) return
  for (const [baseSlug, members] of previousState.baseMemberSlugs) {
    if (members.has(targetSlug) && currentPlans.has(baseSlug)) {
      slugsToRebuild.add(baseSlug)
    }
  }
}

const addBasesWithCodeDependency = (
  path: FilePath,
  slugsToRebuild: Set<FullSlug>,
  currentPlans: Map<FullSlug, BaseRenderPlan>,
) => {
  for (const [baseSlug, plan] of currentPlans) {
    const deps = (plan.baseData.codeDependencies as string[] | undefined) ?? []
    if (deps.includes(path)) {
      slugsToRebuild.add(baseSlug)
    }
  }
}

const addBasesLinkingToSlug = (
  targetSlug: string,
  slugsToRebuild: Set<FullSlug>,
  currentPlans: Map<FullSlug, BaseRenderPlan>,
  nextState: BaseViewPartialState,
  previousState: BaseViewPartialState | undefined,
) => {
  for (const sourceSlug of previousState?.incomingLinksBySlug.get(targetSlug) ?? []) {
    addBasesContainingSlug(sourceSlug, slugsToRebuild, currentPlans, previousState)
  }
  for (const sourceSlug of nextState.incomingLinksBySlug.get(targetSlug) ?? []) {
    addBasesContainingSlug(sourceSlug, slugsToRebuild, currentPlans, previousState)
  }
}

export function planBaseViewPartialEmit(
  content: ProcessedContent[],
  changeEvents: ChangeEvent[],
  previousState?: BaseViewPartialState,
): BaseViewPartialPlan {
  const allFiles = content.map(([, file]) => file.data)
  const baseFilesBySlug = collectBaseFiles(content)
  const basePlans = collectBaseRenderPlans(baseFilesBySlug, allFiles)
  const nextState = collectBaseViewPartialState(allFiles, basePlans)
  const slugsToRebuild = new Set<FullSlug>()

  if (!previousState && changeEvents.length > 0) {
    for (const slug of basePlans.keys()) {
      slugsToRebuild.add(slug)
    }
    return { allFiles, basePlans, nextState, slugsToRebuild }
  }

  for (const changeEvent of changeEvents) {
    const data = changeEvent.file?.data
    const baseSlug = data ? baseSlugForData(data) : undefined
    if (baseSlug) {
      if (basePlans.has(baseSlug)) {
        slugsToRebuild.add(baseSlug)
      }
      addBasesWithCodeDependency(changeEvent.path, slugsToRebuild, basePlans)
      continue
    }

    const changedSlug = data
      ? slugKeyForData(data)
      : (previousState?.slugByPath.get(changeEvent.path) ?? slugKeyForPath(changeEvent.path))

    if (changedSlug) {
      addBasesContainingSlug(changedSlug, slugsToRebuild, basePlans, previousState)
      addBasesLinkingToSlug(changedSlug, slugsToRebuild, basePlans, nextState, previousState)

      const previousLinks = previousState?.linksBySlug.get(changedSlug)
      const nextLinks =
        changeEvent.type === 'delete' || !data ? new Set<string>() : linksForData(data)
      if (!sameSet(previousLinks, nextLinks)) {
        for (const link of previousLinks ?? []) {
          addBasesContainingSlug(link, slugsToRebuild, basePlans, previousState)
        }
        for (const link of nextLinks) {
          addBasesContainingSlug(link, slugsToRebuild, basePlans, previousState)
        }
      }
    }

    addBasesWithCodeDependency(changeEvent.path, slugsToRebuild, basePlans)
  }

  return { allFiles, basePlans, nextState, slugsToRebuild }
}
