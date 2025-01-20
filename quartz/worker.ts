import sourceMapSupport from "source-map-support"
sourceMapSupport.install(options)
import cfg from "../quartz.config"
import { Argv, BuildCtx } from "./util/ctx"
import { FilePath, FullSlug } from "./util/path"
import {
  createMarkdownParser,
  createHtmlParser,
  createHtmlProcessor,
  createMarkdownProcessor,
} from "./processors/parse"
import { options } from "./util/sourcemap"
import { MarkdownContent } from "./plugins/vfile"

// only called from worker thread
export async function parseMarkdown(buildId: string, argv: Argv, fps: FilePath[]) {
  // this is a hack
  // we assume markdown parsers can add to `allSlugs`,
  // but don't actually use them
  const allSlugs: FullSlug[] = []
  const allAssets: FullSlug[] = []
  const ctx: BuildCtx = {
    buildId,
    cfg,
    argv,
    allSlugs,
    allAssets,
  }
  const processor = createMarkdownProcessor(ctx)
  const parse = createMarkdownParser(ctx, fps)
  return [await parse(processor), allSlugs]
}

export function parseHtml(
  buildId: string,
  argv: Argv,
  fps: MarkdownContent[],
  allSlugs: FullSlug[],
  allAssets: string[],
) {
  const ctx: BuildCtx = {
    buildId,
    cfg,
    argv,
    allSlugs,
    allAssets,
  }
  const processor = createHtmlProcessor(ctx)
  const parse = createHtmlParser(ctx, fps)
  return parse(processor)
}
