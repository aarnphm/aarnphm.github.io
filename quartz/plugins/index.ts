import { FilePath, FullSlug } from '../util/path'

export * from './transformers'
export * from './filters'
export * from './emitters'
export { getStaticResourcesFromPlugins } from '../util/static-resources'

declare module 'vfile' {
  // inserted in processors.ts
  interface DataMap {
    slug: FullSlug
    filePath: FilePath
    relativePath: FilePath
  }
}
