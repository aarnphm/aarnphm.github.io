import type { ContentDetails } from '../../plugins'
import type { LexicalDocument } from './lexical-search-types'
import { pathSearchText } from '../../util/search-text'

export function prepareLexicalDocument(
  id: number,
  slug: string,
  fileData: ContentDetails,
): LexicalDocument {
  return {
    id,
    slug,
    title: fileData.title,
    fileName: pathSearchText(fileData.fileName),
    content: fileData.content ?? '',
    tags: fileData.tags,
    aliases: fileData.aliases,
    target: '',
    protected: fileData.protected === true,
  }
}
