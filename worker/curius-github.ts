import { fromHtml } from 'hast-util-from-html'
import { parseGithubRepositoryUrl } from '../quartz/util/github-embed'
import {
  findElement,
  findElements,
  hasClass,
  metaContent,
  normalizeText,
  textContent,
} from '../quartz/util/hast-query'
import { truncateText } from '../quartz/util/preview'
import { validateArenaReaderTarget } from './arena-reader-resources'

export interface GithubRepositoryCard {
  title: string
  imageUrl?: string
  imageWidth?: number
  imageHeight?: number
  paragraphs: string[]
}

export function readGithubRepositoryCard(
  html: string,
  finalUrl: string,
): GithubRepositoryCard | null {
  const repository = parseGithubRepositoryUrl(finalUrl)
  if (!repository) return null
  const root = fromHtml(html)
  const title = `${repository.owner}/${repository.repo}`
  const identity = metaContent(root, 'octolytics-dimension-repository_nwo')
  if (identity?.toLowerCase() !== title.toLowerCase()) return null

  const image = metaContent(root, 'og:image')
  const imageUrl =
    image?.startsWith('https:') && validateArenaReaderTarget(image) ? image : undefined
  const width = Number(metaContent(root, 'og:image:width'))
  const height = Number(metaContent(root, 'og:image:height'))
  const dimensions = [width, height].every(
    value => Number.isInteger(value) && value > 0 && value <= 10_000,
  )
    ? { imageWidth: width, imageHeight: height }
    : {}
  const readme = findElement(
    root,
    element => element.tagName === 'article' && hasClass(element, 'markdown-body'),
  )
  const paragraphs = readme
    ? findElements(readme, element => element.tagName === 'p')
        .map(element => normalizeText(textContent(element)))
        .filter(Boolean)
        .slice(0, 1)
        .map(text => truncateText(text, 1200))
    : []
  const description = metaContent(root, 'og:description')
  if (paragraphs.length === 0 && description && !description.startsWith('Contribute to '))
    paragraphs.push(truncateText(normalizeText(description), 800))

  return imageUrl || paragraphs.length > 0 ? { title, imageUrl, paragraphs, ...dimensions } : null
}
