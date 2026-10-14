#!/usr/bin/env node
import workerpool from 'workerpool'
const cacheFile = './.quartz-cache/transpiled-worker.mjs'
const { parseFiles, parseMarkdown, processHtml } = await import(cacheFile)
workerpool.worker({ parseFiles, parseMarkdown, processHtml })
