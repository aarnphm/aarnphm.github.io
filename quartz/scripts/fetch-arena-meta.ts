#!/usr/bin/env node
import fs from 'fs'
import path from 'path'

interface ArenaEntry {
  url: string
  description: string
  date: string
  tags: string[]
  lineNumber: number
}

interface MetaInfo {
  title?: string
  description?: string
  ogType?: string
  ogTitle?: string
  ogDescription?: string
  twitterCard?: string
  keywords?: string[]
}

interface ProcessedEntry extends ArenaEntry {
  metaInfo?: MetaInfo
  suggestedTags: string[]
  error?: string
}

// parse meta tags from HTML content
function parseMetaTags(html: string): MetaInfo {
  const metaInfo: MetaInfo = {}

  // extract title
  const titleMatch = html.match(/<title[^>]*>(.*?)<\/title>/i)
  if (titleMatch) {
    metaInfo.title = titleMatch[1].trim()
  }

  // extract og:title
  const ogTitleMatch = html.match(
    /<meta[^>]*property=["']og:title["'][^>]*content=["']([^"']*)["']/i,
  )
  if (ogTitleMatch) {
    metaInfo.ogTitle = ogTitleMatch[1]
  }

  // extract og:description
  const ogDescMatch = html.match(
    /<meta[^>]*property=["']og:description["'][^>]*content=["']([^"']*)["']/i,
  )
  if (ogDescMatch) {
    metaInfo.ogDescription = ogDescMatch[1]
  }

  // extract og:type
  const ogTypeMatch = html.match(/<meta[^>]*property=["']og:type["'][^>]*content=["']([^"']*)["']/i)
  if (ogTypeMatch) {
    metaInfo.ogType = ogTypeMatch[1]
  }

  // extract description meta tag
  const descMatch = html.match(/<meta[^>]*name=["']description["'][^>]*content=["']([^"']*)["']/i)
  if (descMatch) {
    metaInfo.description = descMatch[1]
  }

  // extract keywords
  const keywordsMatch = html.match(/<meta[^>]*name=["']keywords["'][^>]*content=["']([^"']*)["']/i)
  if (keywordsMatch) {
    metaInfo.keywords = keywordsMatch[1].split(',').map(k => k.trim())
  }

  // extract twitter card type
  const twitterCardMatch = html.match(
    /<meta[^>]*name=["']twitter:card["'][^>]*content=["']([^"']*)["']/i,
  )
  if (twitterCardMatch) {
    metaInfo.twitterCard = twitterCardMatch[1]
  }

  return metaInfo
}

// fetch meta information from URL
async function fetchMetaInfo(url: string): Promise<MetaInfo | null> {
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 10000) // 10s timeout

    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      },
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      console.warn(`Failed to fetch ${url}: ${response.status} ${response.statusText}`)
      return null
    }

    const html = await response.text()
    return parseMetaTags(html)
  } catch (error) {
    if (error instanceof Error) {
      console.warn(`Error fetching ${url}: ${error.message}`)
    }
    return null
  }
}

// infer tags based on meta information and URL
function inferTags(entry: ArenaEntry, metaInfo: MetaInfo | null): string[] {
  const tags = new Set<string>()

  // domain-based tags
  const url = new URL(entry.url)
  const domain = url.hostname

  if (domain.includes('x.com') || domain.includes('twitter.com')) {
    tags.add('social')
    tags.add('twitter')
  } else if (domain.includes('arxiv.org')) {
    tags.add('paper')
    tags.add('academic')
  } else if (domain.includes('github.com')) {
    tags.add('code')
    tags.add('repository')
  } else if (domain.includes('youtube.com') || domain.includes('youtu.be')) {
    tags.add('video')
  } else if (domain.includes('lesswrong.com')) {
    tags.add('rationality')
    tags.add('discourse')
  } else if (domain.includes('substack.com')) {
    tags.add('article')
    tags.add('newsletter')
  }

  if (!metaInfo) {
    return Array.from(tags)
  }

  // og:type based tags
  if (metaInfo.ogType) {
    if (metaInfo.ogType.includes('article')) tags.add('article')
    if (metaInfo.ogType.includes('video')) tags.add('video')
    if (metaInfo.ogType.includes('website')) tags.add('web')
  }

  // twitter card based tags
  if (metaInfo.twitterCard === 'summary_large_image') {
    tags.add('media')
  }

  // keywords based tags
  if (metaInfo.keywords && metaInfo.keywords.length > 0) {
    metaInfo.keywords.forEach(keyword => {
      const normalized = keyword.toLowerCase().trim()
      if (normalized.length > 2 && normalized.length < 20) {
        tags.add(normalized)
      }
    })
  }

  // content analysis from description
  const description = (metaInfo.ogDescription || metaInfo.description || '').toLowerCase()
  const title = (metaInfo.ogTitle || metaInfo.title || '').toLowerCase()
  const content = `${title} ${description}`

  // topic-based inference
  if (content.match(/\b(ai|artificial intelligence|machine learning|ml|llm)\b/)) tags.add('ai')
  if (content.match(/\b(research|study|paper|academic)\b/)) tags.add('research')
  if (content.match(/\b(philosophy|philosophical|ethics|moral)\b/)) tags.add('philosophy')
  if (content.match(/\b(money|capitalism|economy|economic|finance)\b/)) tags.add('economics')
  if (content.match(/\b(love|relationship|dating|romance)\b/)) tags.add('relationships')
  if (content.match(/\b(poem|poetry|creative writing)\b/)) tags.add('poetry')
  if (content.match(/\b(life|living|existence)\b/)) tags.add('life')
  if (content.match(/\b(meme|funny|humor|joke)\b/)) tags.add('humor')

  return Array.from(tags)
}

// parse are.na.md file and extract entries with specific tag
function parseArenaFile(filePath: string, targetTag: string): ArenaEntry[] {
  const content = fs.readFileSync(filePath, 'utf-8')
  const lines = content.split('\n')
  const entries: ArenaEntry[] = []

  let i = 0
  while (i < lines.length) {
    const line = lines[i]

    // check if line starts with "- http" (entry line)
    if (line.trim().startsWith('- http')) {
      const urlMatch = line.match(/^-\s+(https?:\/\/[^\s]+)(?:\s+--\s+(.*))?/)
      if (urlMatch) {
        const url = urlMatch[1]
        const description = urlMatch[2]?.replace(/\*\*$/, '').trim() || ''

        // look ahead for [meta] section
        let date = ''
        let tags: string[] = []
        let j = i + 1

        // check next lines for meta section
        if (j < lines.length && lines[j].trim() === '- [meta]:') {
          j++
          while (j < lines.length && lines[j].match(/^\s{4}-\s+/)) {
            const metaLine = lines[j].trim()

            // extract date
            const dateMatch = metaLine.match(/^-\s+date:\s+(.+)/)
            if (dateMatch) {
              date = dateMatch[1]
            }

            // extract tags
            const tagsMatch = metaLine.match(/^-\s+tags:\s+\[(.*)\]/)
            if (tagsMatch) {
              tags = tagsMatch[1].split(',').map(t => t.trim())
            }

            j++
          }
        }

        // check if entry has target tag
        if (tags.includes(targetTag)) {
          entries.push({
            url,
            description,
            date,
            tags,
            lineNumber: i + 1, // 1-indexed
          })
        }
      }
    }

    i++
  }

  return entries
}

// delay helper
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// main execution
async function main() {
  const args = process.argv.slice(2)
  const targetTag = args.find(arg => arg.startsWith('--tag='))?.split('=')[1] || 'fruit'
  const outputPath =
    args.find(arg => arg.startsWith('--output='))?.split('=')[1] || 'content/are.na-meta.json'
  const delayMs = parseInt(args.find(arg => arg.startsWith('--delay='))?.split('=')[1] || '1500')

  const arenaPath = path.join(process.cwd(), 'content/are.na.md')

  console.log(`Parsing ${arenaPath} for entries tagged with "${targetTag}"...`)
  const entries = parseArenaFile(arenaPath, targetTag)
  console.log(`Found ${entries.length} entries with tag "${targetTag}"`)

  const processedEntries: ProcessedEntry[] = []

  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i]
    console.log(`\n[${i + 1}/${entries.length}] Fetching meta for: ${entry.url}`)

    const metaInfo = await fetchMetaInfo(entry.url)

    if (metaInfo) {
      console.log(`  Title: ${metaInfo.ogTitle || metaInfo.title || 'N/A'}`)
      console.log(`  Type: ${metaInfo.ogType || 'N/A'}`)
    }

    const suggestedTags = inferTags(entry, metaInfo)
    console.log(`  Suggested tags: ${suggestedTags.join(', ')}`)

    processedEntries.push({
      ...entry,
      metaInfo: metaInfo || undefined,
      suggestedTags,
      error: metaInfo ? undefined : 'Failed to fetch meta information',
    })

    // rate limiting
    if (i < entries.length - 1) {
      await delay(delayMs)
    }
  }

  // save results
  const outputFullPath = path.join(process.cwd(), outputPath)
  fs.writeFileSync(outputFullPath, JSON.stringify(processedEntries, null, 2))
  console.log(`\n\nResults saved to ${outputFullPath}`)

  // summary
  const successful = processedEntries.filter(e => !e.error).length
  console.log(`\nSummary:`)
  console.log(`  Total entries: ${processedEntries.length}`)
  console.log(`  Successfully fetched: ${successful}`)
  console.log(`  Failed: ${processedEntries.length - successful}`)
}

main().catch(console.error)
