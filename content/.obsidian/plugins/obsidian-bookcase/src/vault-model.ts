// The only module that calls the garden's vault modules. It hands resolved entries to the garden's
// `buildVault`, swaps the scene's theme rules for Obsidian's, drops the garden's card dialog (the page
// panel takes its place) and serializes the rest.
import type { Element, ElementContent } from 'hast'
import { toHtml } from 'hast-util-to-html'
import {
  type VaultEntry,
  type VaultView,
  buildVault,
} from '../../../../../quartz/util/base/vault/render'
import { VAULT_MATERIALS, vaultCss } from '../../../../../quartz/util/base/vault/styles'

export { VAULT_DEFAULTS, VAULT_LIMITS, lidRibs } from '../../../../../quartz/util/base/vault/layout'
export type { VaultEntry, VaultView }

const isElement = (node: ElementContent): node is Element => node.type === 'element'

const hasClass = (node: Element, name: string) => {
  const cls = node.properties.className
  return Array.isArray(cls) && cls.includes(name)
}

function findScene(node: Element): Element | null {
  if (node.tagName === 'svg' && hasClass(node, 'vault-scene')) return node
  for (const child of node.children)
    if (isElement(child)) {
      const found = findScene(child)
      if (found) return found
    }
  return null
}

/** Markup of the vault for `entries`. */
export function vaultMarkup(entries: VaultEntry[], view: VaultView): string {
  const tree = buildVault(entries, view)
  tree.children = tree.children.filter(node => !(isElement(node) && node.tagName === 'dialog'))
  const scene = findScene(tree)
  if (scene)
    scene.children = scene.children.map(node => {
      if (!isElement(node)) return node
      // Obsidian marks the dark theme on body.
      if (node.tagName === 'style')
        return {
          ...node,
          children: [
            { type: 'text', value: vaultCss(VAULT_MATERIALS, '.vault-scene', '.theme-dark ') },
          ],
        }
      return node
    })
  return toHtml(tree)
}
