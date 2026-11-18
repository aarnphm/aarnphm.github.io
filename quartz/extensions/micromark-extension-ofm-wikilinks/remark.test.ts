/**
 * integration tests for remarkWikilink plugin.
 * verifies full unified pipeline including mdast-to-hast conversion.
 */

import type { Element } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import rehypeStringify from 'rehype-stringify'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { visit } from 'unist-util-visit'
import { parseLiteralReference } from '../../util/references/literal'
import { referenceHastHandler } from '../../util/references/render'
import { referenceFields } from '../../util/references/request'
import { remarkWikilink } from './index'

/**
 * helper to process markdown through full pipeline.
 * parses markdown → converts to HTML → returns HTML string.
 */
function processToHTML(markdown: string): string {
  const processor = unified()
    .use(remarkParse)
    //@ts-ignore
    .use(remarkWikilink)
    .use(remarkRehype, { handlers: { wikilink: referenceHastHandler } })
    .use(rehypeStringify)

  const result = processor.processSync(markdown)
  return result.toString()
}

function linkDestinations(html: string): string[] {
  const destinations: string[] = []
  visit(fromHtml(html, { fragment: true }), 'element', node => {
    if (node.tagName === 'a' && typeof node.properties.href === 'string')
      destinations.push(node.properties.href)
  })
  return destinations
}

function renderedElements(markdown: string): Element[] {
  const processor = unified()
    .use(remarkParse)
    .use(remarkWikilink)
    .use(remarkRehype, { handlers: { wikilink: referenceHastHandler } })
  const tree = processor.runSync(processor.parse(markdown))
  const elements: Element[] = []
  visit(tree, 'element', node => {
    elements.push(node)
  })
  return elements
}

function transclude(markdown: string): Element {
  const node = renderedElements(markdown).find(element => element.tagName === 'blockquote')
  assert(node, 'Reference HAST handler must emit the transclusion host')
  assert(node.data?.referenceRequest, 'Transclusion host retains its typed reference request')
  return node
}

describe('remarkWikilink integration', () => {
  describe('mdast-to-hast conversion', () => {
    test('converts simple wikilink to anchor tag', () => {
      const html = processToHTML('[[test]]')
      assert(html.includes('<a href="test">test</a>'))
    })

    test('converts wikilink with alias', () => {
      const html = processToHTML('[[page|display text]]')
      assert(html.includes('<a href="page">display text</a>'))
    })

    test('converts wikilink with anchor', () => {
      const html = processToHTML('[[page#section]]')
      assert(html.includes('<a href="page#section">page > section</a>'))
    })

    test('converts wikilink with anchor and alias', () => {
      const html = processToHTML('[[page#section|custom]]')
      assert(html.includes('<a href="page#section">custom</a>'))
    })
  })

  describe('absolute paths', () => {
    test('preserves absolute path starting with /', () => {
      const html = processToHTML('[[/tags/ml]]')
      assert(html.includes('<a href="/tags/ml">/tags/ml</a>'))
    })

    test('preserves absolute path with anchor', () => {
      const html = processToHTML('[[/tags/ml#section]]')
      assert(html.includes('<a href="/tags/ml#section">/tags/ml > section</a>'))
    })

    test('preserves absolute path with alias', () => {
      const html = processToHTML('[[/tags/ml|Machine Learning]]')
      assert(html.includes('<a href="/tags/ml">Machine Learning</a>'))
    })

    test('preserves nested absolute paths', () => {
      const html = processToHTML('[[/thoughts/papers/attention]]')
      assert(html.includes('<a href="/thoughts/papers/attention">/thoughts/papers/attention</a>'))
    })

    test('absolute path bypasses slugification', () => {
      const html = processToHTML('[[/Tags With Spaces]]')
      // should preserve exactly as written, no slugification
      assert(html.includes('<a href="/Tags With Spaces">/Tags With Spaces</a>'))
    })
  })

  describe('same-file anchors', () => {
    test('handles same-file anchor with empty target', () => {
      const html = processToHTML('[[#heading]]')
      assert(html.includes('<a href="#heading">heading</a>'))
    })

    test('slugifies same-file anchor during HAST conversion', () => {
      const html = processToHTML('[[#Section Title]]')
      assert(html.includes('<a href="#section-title">Section Title</a>'))
    })

    test('handles same-file anchor with alias', () => {
      const html = processToHTML('[[#heading|Jump Here]]')
      assert(html.includes('<a href="#heading">Jump Here</a>'))
    })

    test('handles block reference', () => {
      const html = processToHTML('[[#^block-id]]')
      assert(html.includes('<a href="#^block-id">block-id</a>'))
    })
  })

  describe('image embeds', () => {
    test('accepts a table-escaped alias after image settings in both parsers', () => {
      const source = '![[photo.webp#{width:10rem}\\|Photo\\|160]]'
      const html = processToHTML(source)
      assert(html.includes('style="width:10rem"'))
      assert(html.includes('alt="Photo"'))
      assert(html.includes('width="160"'))
      const parsed = parseLiteralReference(source)
      assert(parsed)
      const fields = referenceFields(parsed)
      assert.equal(fields.settings, '{width:10rem}')
      assert.equal(fields.label, 'Photo|160')
    })

    test('applies raw CSS declarations without changing the image source, caption or dimensions', () => {
      const declarations =
        'float:right; width: min(40%, 18rem); margin: 0 0 1rem 1rem; object-fit: cover; --ink: #abc; border-color: var(--ink) !important'
      const html = processToHTML(
        `![[thoughts/images/bike-trainer.webp#{${declarations}}|Trainer|600x400]]`,
      )
      const tree = fromHtml(html, { fragment: true })
      let images = 0
      visit(tree, 'element', node => {
        if (node.tagName !== 'img') return
        images++
        assert.equal(node.properties.src, 'thoughts/images/bike-trainer.webp')
        assert.equal(node.properties.style, declarations)
        assert.equal(node.properties.alt, 'Trainer')
        assert.equal(node.properties.width, 600)
        assert.equal(node.properties.height, 400)
      })
      assert.equal(images, 1)
    })

    test('serializes CSS as an attribute even when it contains HTML and quoted URLs', () => {
      const declarations =
        'background-image:url("data:image/svg+xml,<svg xmlns=\'http://www.w3.org/2000/svg\'></svg>"); --label:"<script>bad()</script>"'
      const html = processToHTML(`![[photo.webp#{${declarations}}]]`)
      const tree = fromHtml(html, { fragment: true })
      visit(tree, 'element', node => {
        assert.notEqual(node.tagName, 'script')
        assert.notEqual(node.tagName, 'svg')
        if (node.tagName === 'img') assert.equal(node.properties.style, declarations)
      })
    })

    test('keeps styles scoped to image embeds and retains typed settings', () => {
      const nodes = renderedElements(
        '![[photo.webp#{float:right}]] ![[plain.webp]] [[page#{popover:false}]] ![[note#{collapsed:true}]]',
      )
      for (const node of nodes) {
        if (node.properties.src === 'photo.webp') {
          assert.equal(node.properties.style, 'float:right')
          assert.deepEqual(node.data?.referenceRequest?.resolvedSettings, {
            kind: 'css',
            raw: 'float:right',
          })
        } else assert.equal(node.properties.style, undefined)
        if (node.tagName === 'blockquote') {
          const settings = node.data?.referenceRequest?.resolvedSettings
          assert.equal(settings?.kind, 'object')
          assert(settings?.kind === 'object')
          assert.equal(settings.value.collapsed, true)
        }
        if (node.properties.href === 'page') {
          const settings = node.data?.referenceRequest?.resolvedSettings
          assert(settings?.kind === 'object')
          assert.equal(settings.value.popover, false)
        }
      }
    })

    test('leaves incomplete image settings literal and omits an empty style', () => {
      assert.equal(
        processToHTML('![[photo.webp#{float:right]]'),
        '<p>![[photo.webp#{float:right]]</p>',
      )
      assert(!processToHTML('![[photo.webp#{}]]').includes('style='))
    })

    test('preserves the transcript opt-in through Markdown to HTML', () => {
      const html = processToHTML('![[https://www.youtube.com/watch?v=ebx7itIvobo#transcripts]]')
      assert(html.includes('src="https://www.youtube.com/watch?v=ebx7itIvobo#transcripts"'))
    })

    test('preserves raw YouTube timestamp fragments', () => {
      const html = processToHTML('![[https://youtu.be/ebx7itIvobo#t=35]]')
      assert(html.includes('src="https://youtu.be/ebx7itIvobo#t=35"'))
    })

    test('converts image embed to img tag', () => {
      const html = processToHTML('![[photo.jpg]]')
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('width="auto"'))
      assert(html.includes('height="auto"'))
    })

    test('converts image with caption', () => {
      const html = processToHTML('![[photo.jpg|Beautiful sunset]]')
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('alt="Beautiful sunset"'))
    })

    test('converts image with dimensions', () => {
      const html = processToHTML('![[photo.jpg|400x300]]')
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('width="400"'))
      assert(html.includes('height="300"'))
    })

    test('converts image with caption and dimensions', () => {
      const html = processToHTML('![[photo.jpg|Sunset|400x300]]')
      assert(html.includes('<img src="photo.jpg"'))
      assert(html.includes('width="400"'))
      assert(html.includes('height="300"'))
      assert(html.includes('alt="Sunset"'))
    })

    test('handles various image extensions', () => {
      const extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'webp']
      for (const ext of extensions) {
        const html = processToHTML(`![[image.${ext}]]`)
        assert(html.includes(`<img src="image.${ext}"`), `should handle .${ext}`)
      }
    })
  })

  describe('video embeds', () => {
    test('converts video embed to video tag', () => {
      const html = processToHTML('![[demo.mp4]]')
      assert(html.includes('<video src="demo.mp4"'))
      assert(html.includes('controls'))
      assert(html.includes('loop'))
    })

    test('handles various video extensions', () => {
      const extensions = ['mp4', 'webm', 'ogv', 'mov', 'mkv']
      for (const ext of extensions) {
        const html = processToHTML(`![[video.${ext}]]`)
        assert(html.includes(`<video src="video.${ext}"`), `should handle .${ext}`)
      }
    })
  })

  describe('audio embeds', () => {
    test('converts audio embed to audio tag', () => {
      const html = processToHTML('![[song.mp3]]')
      assert(html.includes('<audio src="song.mp3"'))
      assert(html.includes('controls'))
    })

    test('handles various audio extensions', () => {
      const extensions = ['mp3', 'wav', 'm4a', 'ogg', '3gp', 'flac']
      for (const ext of extensions) {
        const html = processToHTML(`![[audio.${ext}]]`)
        assert(html.includes(`<audio src="audio.${ext}"`), `should handle .${ext}`)
      }
    })

    test('carries alias as caption', () => {
      const html = processToHTML('![[song.mp3|Hey, Who Really Cares—Linda Perhacs]]')
      assert(html.includes('data-caption="Hey, Who Really Cares—Linda Perhacs"'))
    })
  })

  describe('PDF embeds', () => {
    test('converts PDF embed to viewer host', () => {
      const html = processToHTML('![[paper.pdf]]')
      assert(html.includes('<div class="internal-embed pdf-embed"'))
      assert(html.includes('data-pdf-src="paper.pdf"'))
      assert(html.includes('data-pdf-title="paper.pdf"'))
    })

    test('converts PDF embed options to data attributes', () => {
      const html = processToHTML(
        "![[paper.pdf#{page: 3, fit: 'page', scale: '125%', height: 720}|slides]]",
      )
      assert(html.includes('data-pdf-title="slides"'))
      assert(html.includes('data-pdf-page="3"'))
      assert(html.includes('data-pdf-fit="page"'))
      assert(html.includes('data-pdf-scale="1.25"'))
      assert(html.includes('data-pdf-height="720"'))
    })
  })

  describe('block transcludes', () => {
    test('converts transclude to blockquote with typed whole-document selection', () => {
      const node = transclude('![[notes]]')
      assert.deepEqual(node.data?.referenceRequest?.selector, { kind: 'whole' })
      assert(linkDestinations(processToHTML('![[notes]]')).includes('notes'))
    })

    test('converts transclude with anchor', () => {
      const node = transclude('![[notes#summary]]')
      assert.equal(node.data?.referenceRequest?.route?.anchor, '#summary')
      assert(linkDestinations(processToHTML('![[notes#summary]]')).includes('notes#summary'))
    })

    test('preserves raw nested triathlon transclude fragments', () => {
      const node = transclude('![[triathlon#2026-07-26#filter=19471122670&19476629599]]')
      assert.deepEqual(node.data?.referenceRequest?.fragments, [
        '2026-07-26',
        'filter=19471122670&19476629599',
      ])
    })

    test('preserves triathlon trace settings in nested transclude fragments', () => {
      const node = transclude(
        '![[triathlon#2026-08-10#cycling#settings=matched-rides:false&matched-runs:true]]',
      )
      assert.deepEqual(node.data?.referenceRequest?.fragments, [
        '2026-08-10',
        'cycling',
        'settings=matched-rides:false&matched-runs:true',
      ])
    })

    test('preserves analytics settings and sport filters in nested transclude fragments', () => {
      const analytics = transclude(
        '![[triathlon#2026-08-18#analytics#settings=display:simplified&expanded:true]]',
      )
      const cycling = transclude('![[triathlon#2026-08-16#analytics#cycling]]')
      assert.deepEqual(analytics.data?.referenceRequest?.fragments, [
        '2026-08-18',
        'analytics',
        'settings=display:simplified&expanded:true',
      ])
      assert.deepEqual(cycling.data?.referenceRequest?.fragments, [
        '2026-08-16',
        'analytics',
        'cycling',
      ])
    })

    test('converts base view transclude', () => {
      const node = transclude('![[data.base#editors]]')
      assert.equal(node.data?.referenceRequest?.route?.slug, 'data/editors')
      assert(linkDestinations(processToHTML('![[data.base#editors]]')).includes('data/editors'))
    })

    test('converts transclude with block reference', () => {
      const node = transclude('![[notes#^block-id]]')
      assert.equal(node.data?.referenceRequest?.route?.anchor, '#^block-id')
      assert.deepEqual(node.data?.referenceRequest?.fragments, ['^block-id'])
    })

    test('converts absolute transclude with block reference and alias', () => {
      const source = '![[/quotes#^tommy|from tommy]]'
      const node = transclude(source)
      assert.equal(node.data?.referenceRequest?.route?.anchor, '#^tommy')
      assert.equal(node.data?.referenceRequest?.label, 'from tommy')
      assert(linkDestinations(processToHTML(source)).includes('/quotes#^tommy'))
    })

    test('converts notebook cell transclude anchor and alias', () => {
      const node = transclude('![[thoughts/Jax#code-cell-1|example]]')
      assert.equal(node.data?.referenceRequest?.route?.slug, 'thoughts/Jax')
      assert.equal(node.data?.referenceRequest?.route?.anchor, '#code-cell-1')
      assert.equal(node.data?.referenceRequest?.label, 'example')
    })

    test('converts transclude metadata into typed settings', () => {
      const node = transclude('![[thoughts/Attention#{collapsed: true}]]')
      const settings = node.data?.referenceRequest?.resolvedSettings
      assert(settings?.kind === 'object')
      assert.equal(settings.value.collapsed, true)
    })

    test('converts transclude with alias', () => {
      const node = transclude('![[notes#summary|See notes]]')
      assert.equal(node.data?.referenceRequest?.label, 'See notes')
      assert.equal(node.data?.referenceRequest?.route?.anchor, '#summary')
    })
  })

  describe('published route extensions', () => {
    test('strips .md extension from href by default', () => {
      const html = processToHTML('[[notes.md]]')
      // href should strip .md, but display text keeps it
      assert(html.includes('<a href="notes">notes.md</a>'))
    })

    test('strips .base extension from href by default', () => {
      const html = processToHTML('[[data.base]]')
      // href should strip .base, but display text keeps it
      assert(html.includes('<a href="data">data.base</a>'))
    })

    test('maps base view anchors to view slugs', () => {
      const html = processToHTML('[[data.base#editors]]')
      assert(html.includes('<a href="data/editors">editors</a>'))
    })

    test('maps .fc deck links to the flashcards view slug', () => {
      const html = processToHTML('[[thoughts/Sets.fc]]')
      assert(html.includes('<a href="thoughts/Sets/flashcards">thoughts/Sets.fc</a>'))
    })

    test('maps .flashcards deck links to the flashcards view slug', () => {
      const html = processToHTML('[[thoughts/Sets.flashcards]]')
      assert(html.includes('<a href="thoughts/Sets/flashcards">thoughts/Sets.flashcards</a>'))
    })

    test('maps .fc deck transcludes to the flashcards view slug', () => {
      const html = processToHTML('![[thoughts/Sets.fc]]')
      assert.equal(
        transclude('![[thoughts/Sets.fc]]').data?.referenceRequest?.route?.slug,
        'thoughts/Sets/flashcards',
      )
      assert(linkDestinations(html).includes('thoughts/Sets/flashcards'))
    })

    test('maps .fc.md deck links to the flashcards view slug', () => {
      const html = processToHTML('[[thoughts/Sets.fc.md]]')
      assert(html.includes('<a href="thoughts/Sets/flashcards">thoughts/Sets.fc.md</a>'))
    })

    test('maps .flashcards.md deck links to the flashcards view slug', () => {
      const html = processToHTML('[[thoughts/Sets.flashcards.md]]')
      assert(html.includes('<a href="thoughts/Sets/flashcards">thoughts/Sets.flashcards.md</a>'))
    })

    test('maps absolute .fc deck links to the flashcards view slug', () => {
      const html = processToHTML('[[/thoughts/Sets.fc]]')
      assert(html.includes('href="/thoughts/Sets/flashcards"'))
    })

    test('maps absolute .fc.md deck links to the flashcards view slug', () => {
      const html = processToHTML('[[/thoughts/Sets.fc.md]]')
      assert(html.includes('href="/thoughts/Sets/flashcards"'))
    })

    test('preserves resource extensions in published links', () => {
      const html = processToHTML('[[document.pdf]]')
      assert(html.includes('document.pdf'))
    })

    test('case insensitive extension matching', () => {
      const html = processToHTML('[[Notes.MD]]')
      // slugifyFilePath doesn't lowercase - preserves "Notes.MD" as-is
      // but .MD extension is not in default strip list (lowercase .md is)
      assert(html.includes('<a href="Notes.MD">Notes.MD</a>'))
    })
  })

  describe('standalone target interpretation', () => {
    test('slugifies file paths during HAST conversion', () => {
      const html = processToHTML('[[File With Spaces]]')
      // sluggify replaces spaces with dashes but preserves case
      assert(html.includes('href="File-With-Spaces"'))
      assert(html.includes('>File With Spaces</a>'))
    })

    test('handles nested anchor segments', () => {
      const html = processToHTML('[[file#Parent#Child]]')
      assert(html.includes('href="file#child"'))
      const node = renderedElements('[[file#Parent#Child]]').find(
        element => element.tagName === 'a',
      )
      assert.deepEqual(node?.data?.referenceRequest?.fragments, ['Parent', 'Child'])
      assert(html.includes('file > Parent > Child'))
    })

    test('slugifies anchor text', () => {
      const html = processToHTML('[[file#Section Title]]')
      // slugAnchor lowercases and replaces spaces with hyphens
      assert(html.includes('<a href="file#section-title">file > Section Title</a>'))
    })

    test('strips LaTeX from anchors', () => {
      const html = processToHTML('[[file#$ mu$]]')
      assert(html.includes('<a href="file#mu">file > $ mu$</a>'))
    })

    test('combines path slugification and anchor processing', () => {
      const html = processToHTML('[[My Notes.md#Section Title|Custom]]')
      // sluggify preserves case for path, strips .md extension, slugAnchor lowercases anchor
      assert(html.includes('href="My-Notes#section-title"'))
      assert(html.includes('>Custom</a>'))
    })
  })

  describe('complex integration scenarios', () => {
    test('handles wikilinks in lists', () => {
      const html = processToHTML('- Item with [[link]]\n- Another [[link2|alias]]')
      assert(html.includes('<ul>'))
      assert(html.includes('<li>'))
      assert(html.includes('<a href="link">link</a>'))
      assert(html.includes('<a href="link2">alias</a>'))
    })

    test('handles wikilinks in blockquotes', () => {
      const html = processToHTML('> This is [[quoted|a quote]] text')
      assert(html.includes('<blockquote>'))
      assert(html.includes('<a href="quoted">a quote</a>'))
    })

    test('handles wikilinks with surrounding text', () => {
      const html = processToHTML('Before [[link1]] middle [[link2|alias]] after')
      assert(html.includes('Before'))
      assert(html.includes('<a href="link1">link1</a>'))
      assert(html.includes('middle'))
      assert(html.includes('<a href="link2">alias</a>'))
      assert(html.includes('after'))
    })

    test('handles nested structures', () => {
      const html = processToHTML('Paragraph with [[link]].\n\n> Quote with [[another]].')
      assert(html.includes('<a href="link">link</a>'))
      assert(html.includes('<a href="another">another</a>'))
      assert(html.includes('<p>'))
      assert(html.includes('<blockquote>'))
    })
  })

  describe('edge cases', () => {
    test('handles empty wikilink', () => {
      const html = processToHTML('[[]]')
      assert(html.includes('<a href="">'))
    })

    test('handles wikilink with only alias', () => {
      const html = processToHTML('[[|display]]')
      assert(html.includes('<a href="">display</a>'))
    })

    test('handles escaped characters', () => {
      const html = processToHTML('[[file\\|name]]')
      // backslash consumed, pipe acts as alias
      assert(html.includes('>name</a>'))
    })
  })

  describe('performance and scale', () => {
    test('handles many wikilinks in single document', () => {
      const links = Array.from({ length: 50 }, (_, i) => `[[link${i}]]`).join(' ')
      const html = processToHTML(links)
      for (let i = 0; i < 50; i++) {
        assert(html.includes(`<a href="link${i}">link${i}</a>`))
      }
    })

    test('handles very long aliases', () => {
      const longAlias = 'a'.repeat(1000)
      const html = processToHTML(`[[target|${longAlias}]]`)
      assert(html.includes(`<a href="target">${longAlias}</a>`))
    })
  })
})
