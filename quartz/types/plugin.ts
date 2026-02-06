import { PluggableList } from 'unified'
import { VFile } from 'vfile'
import { ProcessedContent } from '../plugins/vfile'
import { BuildCtx } from '../util/ctx'
import { FilePath } from '../util/path'
import { StaticResources } from '../util/resources'
import { QuartzComponent } from './component'

export interface PluginTypes {
  transformers: QuartzTransformerPluginInstance[]
  filters: QuartzFilterPluginInstance[]
  emitters: QuartzEmitterPluginInstance[]
}

type OptionType = object | undefined
type ExternalResourcesFn = (ctx: BuildCtx) => Partial<StaticResources> | undefined
export type QuartzTransformerPlugin<Options extends OptionType = undefined> = (
  opts?: Options,
) => QuartzTransformerPluginInstance
export type QuartzTransformerPluginInstance = {
  name: string
  textTransform?: (ctx: BuildCtx, src: string) => string
  markdownPlugins?: (ctx: BuildCtx) => PluggableList
  htmlPlugins?: (ctx: BuildCtx) => PluggableList
  externalResources?: ExternalResourcesFn
}

export type QuartzFilterPlugin<Options extends OptionType = undefined> = (
  opts?: Options,
) => QuartzFilterPluginInstance
export type QuartzFilterPluginInstance = {
  name: string
  shouldPublish(ctx: BuildCtx, content: ProcessedContent): boolean
}

export type ChangeEvent = { type: 'add' | 'change' | 'delete'; path: FilePath; file?: VFile }

export type QuartzEmitterPlugin<Options extends OptionType = undefined> = (
  opts?: Options,
) => QuartzEmitterPluginInstance
export type QuartzEmitterPluginInstance = {
  name: string
  emit: (
    ctx: BuildCtx,
    content: ProcessedContent[],
    resources: StaticResources,
  ) => Promise<FilePath[]> | AsyncGenerator<FilePath>
  partialEmit?: (
    ctx: BuildCtx,
    content: ProcessedContent[],
    resources: StaticResources,
    changeEvents: ChangeEvent[],
  ) => Promise<FilePath[]> | AsyncGenerator<FilePath> | null
  /**
   * Returns the components (if any) that are used in rendering the page.
   * This helps Quartz optimize the page by only including necessary resources
   * for components that are actually used.
   */
  getQuartzComponents?: (ctx: BuildCtx) => QuartzComponent[]
  externalResources?: ExternalResourcesFn
}

export interface Notes {
  container: HTMLElement
  main: HTMLElement // the scrollable div
  column: HTMLElement // the actual container for all stacks

  active: boolean
  destroy(): void
  getChain(): string

  open(): Promise<boolean>
  add(href: URL): Promise<boolean>
  navigate(url: URL): Promise<boolean>
}
