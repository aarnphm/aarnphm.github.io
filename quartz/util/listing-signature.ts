import { QuartzPluginData } from '../plugins/vfile'

type ListingFrontmatter = NonNullable<QuartzPluginData['frontmatter']>

function listingFrontmatter(frontmatter?: ListingFrontmatter) {
  return {
    title: frontmatter?.title,
    tags: frontmatter?.tags ?? [],
    date: frontmatter?.date,
    published: frontmatter?.published,
    modified: frontmatter?.modified,
    description: frontmatter?.description,
    noindex: frontmatter?.noindex,
    rss: frontmatter?.rss,
  }
}

function searchFrontmatter(frontmatter?: ListingFrontmatter) {
  return {
    title: frontmatter?.title,
    tags: frontmatter?.tags ?? [],
    aliases: frontmatter?.aliases ?? [],
    date: frontmatter?.date,
    published: frontmatter?.published,
    modified: frontmatter?.modified,
    description: frontmatter?.description,
    pageLayout: frontmatter?.pageLayout,
    protected: frontmatter?.protected,
  }
}

export function pageListingSignature(data: QuartzPluginData): string {
  return JSON.stringify({
    slug: data.slug,
    dates: data.dates,
    description: data.description,
    frontmatter: listingFrontmatter(data.frontmatter),
  })
}

export function pageListingChanged(
  current: QuartzPluginData | undefined,
  previous: QuartzPluginData | undefined,
): boolean {
  if (!current || !previous) return true
  return pageListingSignature(current) !== pageListingSignature(previous)
}

export function pageNavigationSignature(data: QuartzPluginData): string {
  return JSON.stringify({
    listing: pageListingSignature(data),
    aliases: data.frontmatter?.aliases ?? [],
    layout: data.frontmatter?.pageLayout,
    links: data.links ?? [],
    protected: data.frontmatter?.protected,
  })
}

export function pageNavigationChanged(
  current: QuartzPluginData | undefined,
  previous: QuartzPluginData | undefined,
): boolean {
  if (!current || !previous) return true
  return pageNavigationSignature(current) !== pageNavigationSignature(previous)
}

export function pageSearchSignature(data: QuartzPluginData): string {
  return JSON.stringify({
    slug: data.slug,
    dates: data.dates,
    description: data.description,
    frontmatter: searchFrontmatter(data.frontmatter),
    links: data.links ?? [],
    readingTime: data.readingTime,
    text: data.text ?? '',
  })
}

export function pageSearchChanged(
  current: QuartzPluginData | undefined,
  previous: QuartzPluginData | undefined,
): boolean {
  if (!current || !previous) return true
  return pageSearchSignature(current) !== pageSearchSignature(previous)
}

export function pageSitemapSignature(data: QuartzPluginData): string {
  const frontmatter = data.frontmatter
  return JSON.stringify({
    slug: data.slug,
    dates: data.dates,
    date: frontmatter?.date,
    published: frontmatter?.published,
    modified: frontmatter?.modified,
  })
}

export function pageSitemapChanged(
  current: QuartzPluginData | undefined,
  previous: QuartzPluginData | undefined,
): boolean {
  if (!current || !previous) return true
  return pageSitemapSignature(current) !== pageSitemapSignature(previous)
}
