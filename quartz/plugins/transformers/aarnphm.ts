import type { ElementContent, Root as HtmlRoot } from 'hast'
import { h } from 'hastscript'
import { Root } from 'mdast'
import { PluggableList } from 'unified'
import { visit } from 'unist-util-visit'
import content from '../../components/styles/signatures.scss'
import { QuartzTransformerPlugin } from '../../types/plugin'
import svgGenerator from '../../util/signatures'
import { isRecord } from '../../util/type-guards'
import { transformCodeblocks, quoteDisplayDashes, type CodeOptions } from './aarnphm/codeblocks'
import { banks, glyphs, buildEmailSignatureAsset } from './aarnphm/signature-data'
export { banks } from './aarnphm/signature-data'
export type { EmailSignatureAsset } from './aarnphm/signature-data'

type Enable<T> = T & { enable: boolean }

type SignatureOptions = Enable<{ text: string; class: string }>

export interface Options {
  code: CodeOptions
  signature: SignatureOptions
}

const defaultOptions: Options = {
  code: { enable: true, poetry: true, quotes: true, sms: true },
  signature: { enable: true, text: 'with love Aaron.', class: 'signature' },
}

export const Aarnphm: QuartzTransformerPlugin<Partial<Options>> = userOpts => {
  const opts = { ...defaultOptions, ...userOpts }

  const maps = svgGenerator(banks, glyphs)

  return {
    name: 'Aarnphm',
    markdownPlugins({ cfg }) {
      const plugins: PluggableList = []

      if (opts.code.enable) {
        plugins.push(() => (tree: Root, file) => {
          const lang =
            (file.data.frontmatter?.lang ?? cfg.configuration.locale)?.split('-')[0] ?? 'en'
          transformCodeblocks(tree, file, opts.code, lang)
        })
      }

      return plugins
    },
    htmlPlugins({ cfg }) {
      const plugins: PluggableList = []
      if (opts.code.enable && opts.code.quotes) plugins.push(() => quoteDisplayDashes)

      if (opts.signature.enable) {
        plugins.push([
          () => async (tree: HtmlRoot, file) => {
            const signatureText = file.data.frontmatter?.signature
            const text =
              typeof signatureText === 'string' && signatureText.trim().length > 0
                ? signatureText
                : opts.signature.text
            const stroke = cfg.configuration.theme.colors.lightMode.dark
            const shouldGenerateEmailSignature = file.data.frontmatter?.email === true
            let replaced = false

            visit(tree, 'element', (element, index, parent) => {
              if (!parent || index == null) return
              if (
                (element.tagName !== 'p' && element.tagName !== 'div') ||
                element.children.length === 0
              )
                return
              const isLiteralSignToken =
                element.children[0]?.type === 'text' &&
                element.children[0].value.trim() === '[^sign]'

              const containsSignFootnoteRef = (): boolean => {
                const scan = (node: ElementContent): boolean => {
                  if (node.type === 'element') {
                    if (node.tagName === 'sup' && node.children.length) {
                      const first = node.children[0]
                      if (first && first.type === 'element' && first.tagName === 'a') {
                        const href =
                          typeof first.properties.href === 'string' ? first.properties.href : ''
                        const id =
                          typeof first.properties.id === 'string' ? first.properties.id : ''
                        if (href.includes('fn-sign') || id.includes('fnref-sign')) return true
                      }
                    }
                    return node.children.some(scan)
                  }
                  return false
                }
                return scan(element)
              }

              if (isLiteralSignToken || containsSignFootnoteRef()) {
                parent.children.splice(index, 1, h(`p.${opts.signature.class}`, maps(text)))
                replaced = true
              }
            })
            if (!shouldGenerateEmailSignature) {
              if (!replaced) return
              if (file.data.emailSignature) delete file.data.emailSignature
              return
            }
            const existing = file.data.emailSignature
            if (
              isRecord(existing) &&
              existing.text === text &&
              typeof existing.content === 'string' &&
              existing.content.length > 0
            )
              return
            const asset = await buildEmailSignatureAsset(text, stroke)
            file.data.emailSignature = asset
          },
        ])
      }

      return plugins
    },
    externalResources: () => {
      return { css: [{ content, spaPreserve: true, inline: true }] }
    },
  }
}
