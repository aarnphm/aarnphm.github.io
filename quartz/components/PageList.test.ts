import assert from 'node:assert'
import test, { describe } from 'node:test'
import { GlobalConfiguration } from '../cfg'
import { QuartzPluginData } from '../plugins/vfile'
import { FullSlug, isFullSlug } from '../util/path'
import { byNaturalSlug, byTitleAlphabetical } from './PageList'

const cfg: Pick<GlobalConfiguration, 'locale'> = { locale: 'en-US' }

function fullSlug(input: string): FullSlug {
  assert(isFullSlug(input))
  return input
}

function page(slug: string, title: string): QuartzPluginData {
  return { slug: fullSlug(slug), frontmatter: { title, pageLayout: 'default' } }
}

describe('page list sorting', () => {
  test('sorts naturally by slug', () => {
    const pages = [
      page('course/99-lab-10', '99 Lab 10'),
      page('course/03-analysis-of-context-free-languages', 'Analysis of Context-Free Languages'),
      page('course/00-notebooks-on-compiler-construction', 'Notebooks on Compiler Construction'),
      page('course/99-lab-2', '99 Lab 2'),
    ]

    assert.deepStrictEqual(
      pages.sort(byNaturalSlug(cfg)).map(file => file.frontmatter?.title),
      [
        'Notebooks on Compiler Construction',
        'Analysis of Context-Free Languages',
        '99 Lab 2',
        '99 Lab 10',
      ],
    )
  })

  test('sorts alphabetically by display title', () => {
    const pages = [
      page('course/00-zed', 'Zed'),
      page('course/99-alpha', 'Alpha'),
      page('course/02-beta', 'Beta'),
    ]

    assert.deepStrictEqual(
      pages.sort(byTitleAlphabetical(cfg)).map(file => file.frontmatter?.title),
      ['Alpha', 'Beta', 'Zed'],
    )
  })
})
