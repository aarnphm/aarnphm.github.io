import FlexSearch, { DocumentData } from "flexsearch"
import { FullSlug, FilePath } from "../../util/path"
import { encode } from "./util"

export interface SearchItem extends DocumentData {
  id: number
  slug: FullSlug
  name: FilePath
  title: string
  content: string
  aliases: string[]
  [key: string]: any
}

const searchIndex = new FlexSearch.Document<SearchItem>({
  encode,
  document: {
    id: "id",
    tag: "slug",
    index: [
      {
        field: "title",
        tokenize: "forward",
      },
      {
        field: "name",
        tokenize: "forward",
      },
      {
        field: "aliases",
        tokenize: "forward",
      },
      {
        field: "content",
        tokenize: "forward",
      },
    ],
  },
})

const itemsById = new Map<number, SearchItem>()
let isPopulated = false

export async function populateSearchIndex(data: ContentIndex): Promise<void> {
  if (isPopulated) return

  let id = 0
  const promises = []

  for (const [slug, fileData] of Object.entries(data)) {
    const item: SearchItem = {
      id,
      slug: slug as FullSlug,
      name: fileData.fileName,
      title: fileData.title ?? "",
      content: fileData.content ?? "",
      aliases: fileData.aliases,
    }

    itemsById.set(id, item)
    promises.push(searchIndex.addAsync(id, item))
    id++
  }

  await Promise.all(promises)
  isPopulated = true
}

export async function querySearchIndex(query: string, limit: number = 10): Promise<SearchItem[]> {
  if (!query || query.trim() === "") {
    return [...itemsById.values()].slice(0, limit)
  }

  const results = await searchIndex.searchAsync({
    query,
    limit,
    index: ["title", "name", "aliases"],
  })

  const allIds = new Set<number>()
  for (const fieldResult of results) {
    for (const id of fieldResult.result as number[]) {
      allIds.add(id)
    }
  }

  return [...allIds]
    .map((id) => itemsById.get(id))
    .filter((item): item is SearchItem => item !== undefined)
    .slice(0, limit)
}
