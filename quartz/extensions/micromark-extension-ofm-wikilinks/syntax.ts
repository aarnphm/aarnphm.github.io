/**
 * micromark syntax extension for Obsidian wikilinks.
 * implements a state machine tokenizer following micromark conventions.
 *
 * syntax supported:
 * - basic: `[[target]]`
 * - with alias: `[[target|alias]]`
 * - with anchor: `[[target#heading]]`
 * - with block ref: `[[target#^block-id]]`
 * - embeds: `![[target]]`
 * - combined: `[[target#anchor|alias]]`
 * - (experimental) metadata support: `[[target#{collapsed: true}|alias]]`
 * - escaped pipes: `[[target\|with\|pipes|alias]]`
 * - escaped hashes: `[[target\#with\#hashes]]`
 *
 * state machine flow:
 * start → (embed?) → openFirst → openSecond → target → (anchor?) → (alias?) → closeFirst → closeSecond → ok
 */

import type { Extension, Tokenizer, State, Code } from "micromark-util-types"
import "./types"

const codes = {
  exclamationMark: 33, // !
  numberSign: 35, // #
  leftCurlyBrace: 123, // {
  rightCurlyBrace: 125, // }
  backslash: 92, // \
  leftSquareBracket: 91, // [
  rightSquareBracket: 93, // ]
  verticalBar: 124, // |
  caret: 94, // ^
}

/**
 * create micromark extension for wikilink syntax.
 */
export function wikilink(): Extension {
  return {
    text: {
      [codes.exclamationMark]: { name: "wikilink", tokenize },
      [codes.leftSquareBracket]: { name: "wikilink", tokenize },
    },
  }
}

/**
 * main wikilink tokenizer.
 * handles state transitions through the wikilink syntax.
 */
const tokenize: Tokenizer = function (this, effects, ok, nok) {
  let previousWasBackslash = false

  return start

  /**
   * start state: check for `!` embed marker or `[` open bracket.
   *
   * ```markdown
   * > | ![[embed]]
   *     ^
   * > | [[link]]
   *     ^
   * ```
   */
  function start(code: Code): State | undefined {
    if (code === codes.exclamationMark) {
      effects.enter("wikilink")
      effects.enter("wikilinkEmbedMarker")
      effects.consume(code)
      effects.exit("wikilinkEmbedMarker")
      return openFirst
    }

    if (code === codes.leftSquareBracket) {
      effects.enter("wikilink")
      return openFirst(code)
    }

    return nok(code)
  }

  /**
   * expect first `[` of opening `[[`.
   *
   * ```markdown
   * > | [[target]]
   *     ^
   * > | ![[embed]]
   *      ^
   * ```
   */
  function openFirst(code: Code): State | undefined {
    if (code !== codes.leftSquareBracket) {
      return nok(code)
    }

    effects.enter("wikilinkOpenMarker")
    effects.consume(code)
    return openSecond
  }

  /**
   * expect second `[` of opening `[[`.
   *
   * ```markdown
   * > | [[target]]
   *      ^
   * ```
   */
  function openSecond(code: Code): State | undefined {
    if (code !== codes.leftSquareBracket) {
      return nok(code)
    }

    effects.consume(code)
    effects.exit("wikilinkOpenMarker")
    return targetStart
  }

  /**
   * start of target component.
   * can immediately transition to anchor, alias, or close if empty.
   *
   * ```markdown
   * > | [[target]]
   *       ^
   * > | [[#anchor]]
   *       ^
   * > | [[|alias]]
   *       ^
   * ```
   */
  function targetStart(code: Code): State | undefined {
    // empty target with anchor: [[#heading]]
    if (code === codes.numberSign) {
      return anchorMarker(code)
    }

    // empty target with alias: [[|alias]]
    if (code === codes.verticalBar) {
      return aliasMarker(code)
    }

    // closing brackets: [[]]
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }

    // start consuming target
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter("wikilinkTarget")
      effects.enter("wikilinkTargetChunk", { contentType: "string" })
      return targetInside(code)
    }

    return nok(code)
  }

  /**
   * inside target text, consuming characters.
   * handles escaping and stops at delimiters.
   * checks for metadata marker `#{` after target.
   *
   * ```markdown
   * > | [[target#anchor]]
   *        ^^^^^^
   * > | [[file\|name|alias]]
   *        ^^^^^^^^^
   * > | [[target#{metadata}]]
   *        ^^^^^^
   * ```
   */
  function targetInside(code: Code): State | undefined {
    // handle backslash escaping
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return targetInside
    }

    // unescaped hash → check for anchor or metadata
    if (code === codes.numberSign && !previousWasBackslash) {
      effects.exit("wikilinkTargetChunk")
      effects.exit("wikilinkTarget")
      previousWasBackslash = false
      return anchorMarker(code)
    }

    // unescaped pipe → alias
    if (code === codes.verticalBar && !previousWasBackslash) {
      effects.exit("wikilinkTargetChunk")
      effects.exit("wikilinkTarget")
      previousWasBackslash = false
      return aliasMarker(code)
    }

    // closing bracket → end
    if (code === codes.rightSquareBracket && !previousWasBackslash) {
      effects.exit("wikilinkTargetChunk")
      effects.exit("wikilinkTarget")
      previousWasBackslash = false
      return closeFirst(code)
    }

    // EOF or special codes → fail
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }

    // consume regular character
    effects.consume(code)
    previousWasBackslash = false
    return targetInside
  }

  /**
   * anchor marker `#`.
   * checks if it's metadata marker `#{` or regular anchor `#`.
   *
   * ```markdown
   * > | [[file#heading]]
   *           ^
   * > | [[file#{metadata}]]
   *           ^
   * ```
   */
  function anchorMarker(code: Code): State | undefined {
    if (code !== codes.numberSign) {
      return nok(code)
    }

    effects.enter("wikilinkAnchorMarker")
    effects.consume(code)
    return anchorMarkerDecide
  }

  /**
   * decide if `#` starts metadata or anchor.
   */
  function anchorMarkerDecide(code: Code): State | undefined {
    // if next char is {, this is metadata marker
    if (code === codes.leftCurlyBrace) {
      effects.exit("wikilinkAnchorMarker")
      effects.enter("wikilinkMetadataMarker")
      effects.consume(code) // consume {
      effects.exit("wikilinkMetadataMarker")
      return metadataStart
    }

    // otherwise, it's an anchor
    effects.exit("wikilinkAnchorMarker")
    return anchorStart(code)
  }

  /**
   * start of anchor text.
   * can be empty, heading text, or block reference `^block-id`.
   *
   * ```markdown
   * > | [[file#heading]]
   *            ^
   * > | [[file#^block]]
   *            ^
   * ```
   */
  function anchorStart(code: Code): State | undefined {
    // empty anchor followed by pipe: [[file#|alias]]
    if (code === codes.verticalBar) {
      return aliasMarker(code)
    }

    // empty anchor followed by close: [[file#]]
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }

    // start anchor text
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter("wikilinkAnchor")
      effects.enter("wikilinkAnchorChunk", { contentType: "string" })
      return anchorInside(code)
    }

    return nok(code)
  }

  /**
   * inside anchor text.
   * allows multiple `#` for subheadings, stops at `|` or `]]`.
   * checks for metadata marker `#{` after anchor content.
   *
   * ```markdown
   * > | [[file#heading#subheading|alias]]
   *            ^^^^^^^^^^^^^^^^^^^
   * > | [[file#^block-id]]
   *            ^^^^^^^^^
   * > | [[file#heading#{metadata}]]
   *            ^^^^^^^
   * ```
   */
  function anchorInside(code: Code): State | undefined {
    // handle backslash escaping
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return anchorInside
    }

    // unescaped hash → check for metadata or continue as subheading
    if (code === codes.numberSign && !previousWasBackslash) {
      effects.exit("wikilinkAnchorChunk")
      effects.exit("wikilinkAnchor")
      previousWasBackslash = false
      return anchorOrMetadata(code)
    }

    // unescaped pipe → alias
    if (code === codes.verticalBar && !previousWasBackslash) {
      effects.exit("wikilinkAnchorChunk")
      effects.exit("wikilinkAnchor")
      previousWasBackslash = false
      return aliasMarker(code)
    }

    // closing bracket → end
    if (code === codes.rightSquareBracket && !previousWasBackslash) {
      effects.exit("wikilinkAnchorChunk")
      effects.exit("wikilinkAnchor")
      previousWasBackslash = false
      return closeFirst(code)
    }

    // EOF or special codes → fail
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }

    // consume character (including additional # for subheadings)
    effects.consume(code)
    previousWasBackslash = false
    return anchorInside
  }

  /**
   * check if `#` after anchor is metadata marker `#{` or subheading `#`.
   *
   * ```markdown
   * > | [[file#heading#{metadata}]]
   *                    ^
   * > | [[file#heading#subheading]]
   *                    ^
   * ```
   */
  function anchorOrMetadata(code: Code): State | undefined {
    if (code !== codes.numberSign) {
      return nok(code)
    }

    effects.consume(code) // consume #
    return anchorOrMetadataDecide
  }

  /**
   * decide if next char starts metadata `{` or continues as subheading.
   */
  function anchorOrMetadataDecide(code: Code): State | undefined {
    // if next char is {, this is metadata marker
    if (code === codes.leftCurlyBrace) {
      effects.enter("wikilinkMetadataMarker")
      effects.consume(code) // consume {
      effects.exit("wikilinkMetadataMarker")
      return metadataStart
    }

    if (code === codes.numberSign) {
      effects.consume(code)
      return anchorOrMetadataDecide
    }

    // otherwise, it's a subheading - continue anchor parsing
    effects.enter("wikilinkAnchor")
    effects.enter("wikilinkAnchorChunk", { contentType: "string" })
    return anchorInside(code)
  }

  /**
   * start of metadata content after `#{`.
   * can be empty, or contain nested braces.
   *
   * ```markdown
   * > | [[target#{key:value}]]
   *              ^
   * > | [[target#{}]]
   *              ^
   * ```
   */
  function metadataStart(code: Code): State | undefined {
    // empty metadata: #{}
    if (code === codes.rightCurlyBrace) {
      return metadataEnd(code)
    }

    // empty metadata followed by alias: #{}|alias
    if (code === codes.verticalBar) {
      return aliasMarker(code)
    }

    // empty metadata at end: #{}]]
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }

    // start metadata content
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter("wikilinkMetadata")
      effects.enter("wikilinkMetadataChunk", { contentType: "string" })
      return metadataInside(code, 1) // start with depth 1 (already consumed opening {)
    }

    return nok(code)
  }

  /**
   * inside metadata content.
   * tracks brace depth to handle nested structures like {a:{b:1}}.
   * handles escaping of braces with backslash.
   *
   * ```markdown
   * > | [[target#{key:value}]]
   *              ^^^^^^^^^^
   * > | [[target#{a:{b:1}}]]
   *              ^^^^^^^^^
   * ```
   */
  function metadataInside(code: Code, braceDepth: number): State | undefined {
    // handle backslash escaping
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return (nextCode: Code) => metadataInside(nextCode, braceDepth)
    }

    // unescaped opening brace → increase depth
    if (code === codes.leftCurlyBrace && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = false
      return (nextCode: Code) => metadataInside(nextCode, braceDepth + 1)
    }

    // unescaped closing brace
    if (code === codes.rightCurlyBrace && !previousWasBackslash) {
      if (braceDepth === 1) {
        // reached end of metadata
        effects.exit("wikilinkMetadataChunk")
        effects.exit("wikilinkMetadata")
        previousWasBackslash = false
        return metadataEnd(code)
      }

      // nested brace, continue consuming
      effects.consume(code)
      previousWasBackslash = false
      return (nextCode: Code) => metadataInside(nextCode, braceDepth - 1)
    }

    // EOF or special codes → fail
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }

    // consume regular character
    effects.consume(code)
    previousWasBackslash = false
    return (nextCode: Code) => metadataInside(nextCode, braceDepth)
  }

  /**
   * after closing `}` of metadata.
   * can be followed by alias or closing brackets.
   *
   * ```markdown
   * > | [[target#{metadata}|alias]]
   *                       ^
   * > | [[target#{metadata}]]
   *                       ^
   * ```
   */
  function metadataEnd(code: Code): State | undefined {
    if (code !== codes.rightCurlyBrace) {
      return nok(code)
    }

    effects.consume(code) // consume closing }

    // peek at next character
    return (nextCode: Code) => {
      // next could be alias or close
      if (nextCode === codes.verticalBar) {
        return aliasMarker(nextCode)
      }

      if (nextCode === codes.rightSquareBracket) {
        return closeFirst(nextCode)
      }

      return nok(nextCode)
    }
  }

  /**
   * alias marker `|`.
   *
   * ```markdown
   * > | [[target|alias]]
   *            ^
   * ```
   */
  function aliasMarker(code: Code): State | undefined {
    if (code !== codes.verticalBar) {
      return nok(code)
    }

    effects.enter("wikilinkAliasMarker")
    effects.consume(code)
    effects.exit("wikilinkAliasMarker")
    return aliasStart
  }

  /**
   * start of alias text.
   * alias can be empty.
   *
   * ```markdown
   * > | [[target|display text]]
   *             ^
   * > | [[target|]]
   *             ^
   * ```
   */
  function aliasStart(code: Code): State | undefined {
    // empty alias: [[target|]]
    if (code === codes.rightSquareBracket) {
      return closeFirst(code)
    }

    // start alias text
    if (code !== null && code !== -5 && code !== -4 && code !== -3) {
      effects.enter("wikilinkAlias")
      effects.enter("wikilinkAliasChunk", { contentType: "string" })
      return aliasInside(code)
    }

    return nok(code)
  }

  /**
   * inside alias text.
   * consumes until `]]`, handles escaping.
   *
   * ```markdown
   * > | [[target|display text]]
   *             ^^^^^^^^^^^^
   * ```
   */
  function aliasInside(code: Code): State | undefined {
    // handle backslash escaping
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return aliasInside
    }

    // closing bracket → end
    if (code === codes.rightSquareBracket && !previousWasBackslash) {
      effects.exit("wikilinkAliasChunk")
      effects.exit("wikilinkAlias")
      previousWasBackslash = false
      return closeFirst(code)
    }

    // EOF or special codes → fail
    if (code === null || code === -5 || code === -4 || code === -3) {
      return nok(code)
    }

    // consume character (aliases can contain pipes, hashes, etc.)
    effects.consume(code)
    previousWasBackslash = false
    return aliasInside
  }

  /**
   * first `]` of closing `]]`.
   *
   * ```markdown
   * > | [[target]]
   *             ^
   * ```
   */
  function closeFirst(code: Code): State | undefined {
    if (code !== codes.rightSquareBracket) {
      return nok(code)
    }

    effects.enter("wikilinkCloseMarker")
    effects.consume(code)
    return closeSecond
  }

  /**
   * second `]` of closing `]]`.
   *
   * ```markdown
   * > | [[target]]
   *              ^
   * ```
   */
  function closeSecond(code: Code): State | undefined {
    if (code !== codes.rightSquareBracket) {
      return nok(code)
    }

    effects.consume(code)
    effects.exit("wikilinkCloseMarker")
    effects.exit("wikilink")
    return ok
  }
}
