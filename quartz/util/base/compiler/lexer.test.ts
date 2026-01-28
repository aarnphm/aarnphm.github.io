import assert from "node:assert"
import test from "node:test"
import { lex } from "./lexer"

test("lexes bracket access with hyphenated keys", () => {
  const result = lex('note["my-field"]')
  const types = result.tokens.map((token) => token.type)
  assert.deepStrictEqual(types, ["identifier", "punctuation", "string", "punctuation", "eof"])
  const value = result.tokens[2]
  if (value.type !== "string") {
    throw new Error("expected string token")
  }
  assert.strictEqual(value.value, "my-field")
})

test("lexes bracket access with escaped quotes", () => {
  const result = lex('note["my\\\\\\"field"]')
  const value = result.tokens.find((token) => token.type === "string")
  if (!value || value.type !== "string") {
    throw new Error("expected string token")
  }
  assert.strictEqual(value.value, 'my"field')
})

test("lexes regex literals with flags", () => {
  const result = lex('name.replace(/:/g, "-")')
  const regexToken = result.tokens.find((token) => token.type === "regex")
  if (!regexToken || regexToken.type !== "regex") {
    throw new Error("expected regex token")
  }
  assert.strictEqual(regexToken.pattern, ":")
  assert.strictEqual(regexToken.flags, "g")
})

test("lexes regex literals with escaped slashes", () => {
  const result = lex("path.matches(/\\\\/)")
  const regexToken = result.tokens.find((token) => token.type === "regex")
  if (!regexToken || regexToken.type !== "regex") {
    throw new Error("expected regex token")
  }
  assert.strictEqual(regexToken.pattern, "\\/")
  assert.strictEqual(regexToken.flags, "")
})

test("lexes division as operator, not regex", () => {
  const result = lex("a / b")
  const operatorToken = result.tokens.find(
    (token) => token.type === "operator" && token.value === "/",
  )
  assert.ok(operatorToken)
  const regexToken = result.tokens.find((token) => token.type === "regex")
  assert.strictEqual(regexToken, undefined)
})
