/**
 * token types for wikilink micromark extension.
 * follows micromark conventions: camelCase names, hierarchical structure.
 */

/**
 * container token for entire wikilink construct.
 * example: `[[target#anchor|alias]]` or `![[embed]]`
 */
export type WikilinkToken = "wikilink"

/**
 * embed prefix `!` before opening brackets.
 * example: `!` in `![[image.png]]`
 */
export type WikilinkEmbedMarker = "wikilinkEmbedMarker"

/**
 * opening `[[` markers.
 */
export type WikilinkOpenMarker = "wikilinkOpenMarker"

/**
 * closing `]]` markers.
 */
export type WikilinkCloseMarker = "wikilinkCloseMarker"

/**
 * target file path component.
 * example: `target` in `[[target#anchor|alias]]`
 */
export type WikilinkTarget = "wikilinkTarget"

/**
 * anchor marker `#` before heading/block ref.
 */
export type WikilinkAnchorMarker = "wikilinkAnchorMarker"

/**
 * anchor text after `#` marker.
 * example: `heading` in `[[file#heading]]`
 * example: `^block-id` in `[[file#^block-id]]`
 */
export type WikilinkAnchor = "wikilinkAnchor"

/**
 * alias marker `|` before display text.
 */
export type WikilinkAliasMarker = "wikilinkAliasMarker"

/**
 * alias display text after `|` marker.
 * example: `display` in `[[target|display]]`
 */
export type WikilinkAlias = "wikilinkAlias"

/**
 * text chunk inside components (used by micromark internals).
 */
export type WikilinkChunk = "wikilinkTargetChunk" | "wikilinkAnchorChunk" | "wikilinkAliasChunk"

/**
 * union of all wikilink token types.
 */
export type WikilinkTokenType =
  | WikilinkToken
  | WikilinkEmbedMarker
  | WikilinkOpenMarker
  | WikilinkCloseMarker
  | WikilinkTarget
  | WikilinkAnchorMarker
  | WikilinkAnchor
  | WikilinkAliasMarker
  | WikilinkAlias
  | WikilinkChunk

declare module "micromark-util-types" {
  interface TokenTypeMap {
    wikilink: "wikilink"
    wikilinkEmbedMarker: "wikilinkEmbedMarker"
    wikilinkOpenMarker: "wikilinkOpenMarker"
    wikilinkCloseMarker: "wikilinkCloseMarker"
    wikilinkTarget: "wikilinkTarget"
    wikilinkTargetChunk: "wikilinkTargetChunk"
    wikilinkAnchorMarker: "wikilinkAnchorMarker"
    wikilinkAnchor: "wikilinkAnchor"
    wikilinkAnchorChunk: "wikilinkAnchorChunk"
    wikilinkAliasMarker: "wikilinkAliasMarker"
    wikilinkAlias: "wikilinkAlias"
    wikilinkAliasChunk: "wikilinkAliasChunk"
  }
}
