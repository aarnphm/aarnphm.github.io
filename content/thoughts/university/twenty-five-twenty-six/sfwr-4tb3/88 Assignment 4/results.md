---
date: '2026-02-10'
description: conversion, parser
id: results
modified: 2026-02-10 21:25:16 GMT-05:00
tags:
  - sfwr4tb3
  - assignment
title: regex equivalence
---

## A1

For each of the following equalities, state either that it is true or give a counterexample if it is false!

1. `(a | b)* a (a | b)* = (a | b)*`

   **false.** counterexample: $\varepsilon$. the LHS requires at least one $a$ (the middle $a$ is mandatory), so it cannot produce $\varepsilon$ or any string of pure $b$'s. the RHS accepts both.

2. `a* (b a*)* = (a | b)*`

   **true.** any string $w \in \{a,b\}^*$ decomposes as: leading $a$'s (matching $a^*$), then repeated segments of a $b$ followed by $a$'s (each matching $ba^*$). conversely, LHS only produces strings over $\{a,b\}$.

3. `(a b | a)* = a (b a)*`

   **false.** counterexample: $\varepsilon$. the LHS accepts $\varepsilon$ (zero iterations), while the RHS mandates a leading $a$. additionally, LHS accepts $aa$ (two iterations of the $a$ alternative), while RHS cannot produce $aa$ since $a(ba)^*$ yields strings of the form $a, aba, ababa, \ldots$

4. `(a | ε) b* = b* | a b*`

   **true.** LHS $= \{\varepsilon, a\} \cdot \{b\}^* = \{b\}^* \cup a\{b\}^* =$ RHS.

5. `(a* b)* a* = (a | b)*`

   **true.** any string over $\{a,b\}$ decomposes as segments of $a$'s each terminated by a $b$ (matching $(a^* b)$), followed by trailing $a$'s (matching $a^*$). if $w$ contains no $b$, it matches $(a^* b)^0 \cdot a^* = a^*$.

6. `(a* b a*)* = (a | b)*`

   **false.** counterexample: $a$. with zero iterations, LHS yields only $\varepsilon$. with $\geq 1$ iterations, each factor $a^* b \; a^*$ contributes at least one $b$. so LHS $= \{\varepsilon\} \cup \{w \in \{a,b\}^* \mid w \text{ contains at least one } b\}$. strings consisting solely of $a$'s (length $\geq 1$) are excluded.

## A2

Visa card numbers start with a 4. New Visa cards have 16 digits, and old cards have 13 digits. MasterCard numbers start with the numbers 51 through 55. All have 16 digits. American Express card numbers start with 34 or 37 and have 15 digits. The last digit is a checksum calculated using Luhn's algorithm; see http://en.wikipedia.org/wiki/Credit_card_number. With web apps in mind, use JavaScript regular expressions to check if the credit card is well-formed and of proper length. Implement Luhn's algorithm in JavaScript.

The Jupyter (IPython) cell magic javascript allows JavaScript to be embedded and executed. Complete the JavaScript functions isValidCreditCard and luhnCheckSum!

You can use `console.log(...)` to output to your web browser's JavaScript console for debugging. The expected output below is one line with Valid! and one line with Invalid! Your implementation will be tested on additional credit card numbers.

```js
function isValidCreditCard(sText) {
  var reVisa = /^4\d{12}(\d{3})?$/
  var reMasterCard = /^5[1-5]\d{14}$/
  var reAmericanExpr = /^3[47]\d{13}$/
  if (
    (reMasterCard.test(sText) || reVisa.test(sText) || reAmericanExpr.test(sText)) &&
    luhnCheckSum(sText) === 0
  ) {
    element.append('Valid!')
  } else {
    element.append('Invalid!')
  }
}

function luhnCheckSum(sCardNum) {
  var sum = 0
  var numDigits = sCardNum.length
  var parity = numDigits % 2
  for (var i = 0; i < numDigits; i++) {
    var digit = parseInt(sCardNum[i], 10)
    if (i % 2 === parity) {
      digit *= 2
      if (digit > 9) digit -= 9
    }
    sum += digit
  }
  return sum % 10
}

isValidCreditCard('378282246310005') // American Express
isValidCreditCard('37873449367100') // American Express
isValidCreditCard('5555555555554444') // MasterCard
isValidCreditCard('5105105105105100') // MasterCard

isValidCreditCard('4111111111111111') // Visa
isValidCreditCard('4222222222222') // Visa
var br = document.createElement('br')
element.appendChild(br)

isValidCreditCard('378282246310003')

isValidCreditCard('5555555555554445')
isValidCreditCard('4111111111111114')
isValidCreditCard('3787344936710007')
isValidCreditCard('411111111111111')
```

## A3

### Part A

Replace two spaces at the line end with an HTML line break `<br>`.

```python
import re

match_pattern = r'  $'
replacement_pattern = r'<br>'
```

`  $` matches exactly two space characters anchored at end-of-string. test 2 (`three spaces`) replaces only the last two, leaving one space before `<br>`. test 3 (spaces in the middle) has no two-space suffix, so no match.

### Part B

Paragraphs (blocks of text separated by blank lines) → wrap each in `<p>...</p>`.

```python
import re

match_pattern = r'(.+?)(?:\n\n|\Z)'
replacement_pattern = r'<p>\1</p>'
```

`.+?` lazily matches paragraph content up to the next `\n\n` boundary or end-of-string `\Z`. each match's group 1 gets wrapped in `<p>` tags. the `\n\n` separator is consumed and discarded.

### Part C

Convert markdown links `[text](url)` to HTML `<a href="url">text</a>`.

```python
import re

match_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
replacement_pattern = r'<a href="\2">\1</a>'
```

group 1 captures the link text (everything inside `[...]`), group 2 captures the URL (everything inside `(...)`).

### Part D

Translate markdown `*italics*` and `**bold**` into HTML `<em>` and `<strong>`, with nesting.

```python
import re


def translate(input_string: str) -> str:
  result = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', input_string)
  result = re.sub(r'\*(.+?)\*', r'<em>\1</em>', result)
  return result
```

ordering matters: `**` gets replaced first so that `\*(.+?)\*` doesn't greedily eat bold markers. after bold is resolved, the remaining single `*` pairs become `<em>`.

### Part E

Convert markdown headers (`# ...`, `## ...`, `### ...`) to HTML `<h1>`, `<h2>`, `<h3>` with auto-generated slug IDs (lowercase, spaces → hyphens).

```python
import re


def translate(input_string: str) -> str:
  def header_replace(m):
    level = len(m.group(1))
    text = m.group(2)
    slug = text.lower().replace(' ', '-')
    return f'<h{level} id="{slug}">{text}</h{level}>'

  return re.sub(
    r'^(#{1,6})\s+(.+)$', header_replace, input_string, flags=re.MULTILINE
  )
```

`(#{1,6})` captures the hash marks (determines heading level via `len`), `\s+` eats the whitespace gap, `(.+)` captures the header text. the replacement function computes the slug from the text.
