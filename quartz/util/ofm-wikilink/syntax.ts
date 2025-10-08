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
      [codes.exclamationMark]: {
        name: "wikilink",
        tokenize: tokenizeWikilink,
      },
      [codes.leftSquareBracket]: {
        name: "wikilink",
        tokenize: tokenizeWikilink,
      },
    },
  }
}

/**
 * main wikilink tokenizer.
 * handles state transitions through the wikilink syntax.
 */
const tokenizeWikilink: Tokenizer = function (effects, ok, nok) {
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
   *
   * ```markdown
   * > | [[target#anchor]]
   *        ^^^^^^
   * > | [[file\|name|alias]]
   *        ^^^^^^^^^
   * ```
   */
  function targetInside(code: Code): State | undefined {
    // handle backslash escaping
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return targetInside
    }

    // unescaped hash → anchor
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
   *
   * ```markdown
   * > | [[file#heading]]
   *           ^
   * ```
   */
  function anchorMarker(code: Code): State | undefined {
    if (code !== codes.numberSign) {
      return nok(code)
    }

    effects.enter("wikilinkAnchorMarker")
    effects.consume(code)
    effects.exit("wikilinkAnchorMarker")
    return anchorStart
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
   *
   * ```markdown
   * > | [[file#heading#subheading|alias]]
   *            ^^^^^^^^^^^^^^^^^^^
   * > | [[file#^block-id]]
   *            ^^^^^^^^^
   * ```
   */
  function anchorInside(code: Code): State | undefined {
    // handle backslash escaping
    if (code === codes.backslash && !previousWasBackslash) {
      effects.consume(code)
      previousWasBackslash = true
      return anchorInside
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
