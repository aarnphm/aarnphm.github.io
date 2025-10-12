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
 * metadata marker `#{` before metadata content.
 */
export type WikilinkMetadataMarker = "wikilinkMetadataMarker"

/**
 * metadata content inside braces.
 * example: `{key:value}` in `[[target#{key:value}]]`
 * stored as raw string for downstream parsing.
 */
export type WikilinkMetadata = "wikilinkMetadata"

/**
 * text chunk inside components (used by micromark internals).
 */
export type WikilinkChunk =
  | "wikilinkTargetChunk"
  | "wikilinkAnchorChunk"
  | "wikilinkMetadataChunk"
  | "wikilinkAliasChunk"

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
  | WikilinkMetadataMarker
  | WikilinkMetadata
  | WikilinkAliasMarker
  | WikilinkAlias
  | WikilinkChunk
