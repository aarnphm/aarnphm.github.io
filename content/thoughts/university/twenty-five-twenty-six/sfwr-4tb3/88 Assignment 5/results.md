---
date: "2026-02-10"
description: quotes and parsing
id: results
modified: 2026-02-23 20:53:52 GMT-05:00
tags:
  - sfwr4tb3
  - assignment
title: path, sed
---

## A1

Implement a sanitizer for pathnames using Flex and C. Read from stdin, produce sanitized portable pathname on stdout or error on stderr.

Rules:

- Components separated by `/`, consecutive `/` same as single `/`
- Final `/` has no meaning, initial `/` is significant
- Leading/trailing spaces allowed but no significance
- Component: `a-z`, `A-Z`, `0-9`, `.` (dot)
- `.` component = current directory (remove), `..` = parent directory (resolve)
- Max 14 chars per component, max 255 chars total pathname

| stdin                | stdout      |
| :------------------- | :---------- |
| `/aaa//bb/c/`        | `/aaa/bb/c` |
| `aaa/b.b/../cc/./dd` | `aaa/cc/dd` |

Errors (to stderr, terminate immediately):

| stdin                            | stderr               |
| :------------------------------- | :------------------- |
| `/a//b/#/c`                      | `invalid character`  |
| `/012345678901234/bb`            | `component too long` |
| `aa/../..`                       | `malformed pathname` |
| `/this/is/a/path/...too long...` | `pathname too long`  |

```text
%%writefile spn.l
%option noyywrap
%{
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PATH 255
#define MAX_COMP 14
#define MAX_DEPTH 128

char path[MAX_PATH + 1];
char *comps[MAX_DEPTH];
int depth;
bool absolute;
int rawlen;

void error(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

char *dupn(const char *s, int n) {
    char *d = malloc(n + 1);
    memcpy(d, s, n);
    d[n] = '\0';
    return d;
}

void push_comp(const char *c, int len) {
    if (len > MAX_COMP) error("component too long");
    if (len == 1 && c[0] == '.') return;
    if (len == 2 && c[0] == '.' && c[1] == '.') {
        if (depth == 0) error("malformed pathname");
        free(comps[depth - 1]);
        depth--;
        return;
    }
    comps[depth++] = dupn(c, len);
}

void output_path() {
    int pos = 0;
    if (absolute) { path[pos++] = '/'; }
    for (int i = 0; i < depth; i++) {
        if (i > 0) { path[pos++] = '/'; }
        int cl = strlen(comps[i]);
        memcpy(path + pos, comps[i], cl);
        pos += cl;
    }
    path[pos] = '\0';
    printf("%s\n", path);
    for (int i = 0; i < depth; i++) free(comps[i]);
    depth = 0;
    absolute = false;
    rawlen = 0;
}
%}

VALID    [a-zA-Z0-9.]
COMP     {VALID}{1,14}
LONGCOMP {VALID}{15,}
INVALID  [^a-zA-Z0-9./\n \t]

%%
^[ \t]*"/"+"/"*  { int i=0; while(yytext[i]==' '||yytext[i]=='\t') i++; rawlen=yyleng-i; absolute=true; }
^[ \t]+          ;
[ \t]+$          ;
"/"+"/"*         { rawlen += yyleng; if (rawlen > MAX_PATH) error("pathname too long"); }
{LONGCOMP}       { error("component too long"); }
{COMP}           { rawlen += yyleng; if (rawlen > MAX_PATH) error("pathname too long"); push_comp(yytext, yyleng); }
{INVALID}        { error("invalid character"); }
\n               { output_path(); }
<<EOF>>          { output_path(); return 0; }
%%

int main() {
    depth = 0;
    absolute = false;
    rawlen = 0;
    yylex();
    return 0;
}
```

## A2

Write a `sed` command that removes double quotation marks surrounding CSV cells that contain neither commas nor double-quotation marks.

Input:

```
"We need to add ""double quotation"" marks around cell contents","in some cases, but,","they can be unnecessary in others."
"Write a sed command",that conservatively removes all "strictly unnecessary" ones,"""and no more."""
Be careful to ignore cells that contain "double quotes with no commas!",test,"test."
```

Expected output:

```
"We need to add ""double quotation"" marks around cell contents","in some cases, but,",they can be unnecessary in others.
Write a sed command,that conservatively removes all "strictly unnecessary" ones,"""and no more."""
Be careful to ignore cells that contain "double quotes with no commas!",test,test.
```

```bash
sed -E ':loop; s/(^|,)"([^",]*)"(,|$)/\1\2\3/g; t loop' excessive-quotes.csv
```

The pattern matches a cell boundary (`^` or `,`), then `"`, then content with no commas or quotes (`[^",]*`), then `"`, then cell boundary (`,` or `$`). The loop ensures all such cells on a line are cleaned.

## A3

Write a `sed` command to convert JSON string-encoded booleans, null, and numbers to their native JSON types.

- `"true"` → `true`, `"false"` → `false`, `"null"` → `null`
- `"<number>"` → `<number>` (integers and decimals, no spaces)

Input: `{"true": true, "false": "false", "null": "null", "number1": "3.14159", ...}`

Expected output: `{"true": true, "false": false, "null": null, "number1": 3.14159, ...}`

```bash
sed -E 's/: "true"/: true/g; s/: "false"/: false/g; s/: "null"/: null/g; s/: "(-?[0-9]+\.?[0-9]*)"/: \1/g' quoted-data.json
```

Replace string-valued `"true"`, `"false"`, `"null"` with their unquoted equivalents. For numbers, match `": "` followed by an optional minus, digits, optional decimal point and more digits, then closing `"`. The number regex `(-?[0-9]+\.?[0-9]*)` avoids matching strings with spaces or letters.

## A4

### Part A: Parse regular expressions into AST

Extend the regex parser from the course notes with attribute evaluation rules so `parse` returns the abstract syntax tree.

Attribute grammar:

```
expression(e)  →  term(e) { '|' term(f) « e := Choice(e, f) » }
term(e)  →  factor(e) { factor(f) « e := Conc(e, f) » }
factor(e) → atom(e) [ '*' « e := Star(e) » | '+' « e := Conc(e, Star(e)) » | '?' « e := Choice(e, ε()) » ]
atom(e)  →  plainchar(e) | escapedchar(e) | '(' expression(e) ')'
plainchar(e)  →  ' ' « e := Sym(' ') » | ... | '~' « e := Sym('~') »
escapedchar(e)  → '\\' ( '(' « e := Sym('(') » | ')' | ... | '|' « e := Sym('|') »)
```

_sol_:

```python
PlainChars = (
  ' !"#$%&\',-./0123456789:;<=>@ABCDEFGHIJKLMNO'
  + 'PQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{}~'
)
EscapedChars = '()*+?\\|'
FirstFactor = PlainChars + '\\('

src: str
pos: int
sym: str


def nxt():
  global pos, sym
  if pos < len(src):
    sym, pos = src[pos], pos + 1
  else:
    sym = chr(0)


def expression():
  e = term()
  while sym == '|':
    nxt()
    f = term()
    e = Choice(e, f)
  return e


def term():
  e = factor()
  while sym in FirstFactor:
    f = factor()
    e = Conc(e, f)
  return e


def factor():
  e = atom()
  if sym == '*':
    nxt()
    e = Star(e)
  elif sym == '+':
    nxt()
    e = Conc(e, Star(e))
  elif sym == '?':
    nxt()
    e = Choice(e, ε())
  return e


def atom():
  if sym in PlainChars:
    e = Sym(sym)
    nxt()
  elif sym == '\\':
    nxt()
    if sym in EscapedChars:
      e = Sym(sym)
      nxt()
    else:
      raise Exception('invalid escaped character at ' + str(pos))
  elif sym == '(':
    nxt()
    e = expression()
    if sym == ')':
      nxt()
    else:
      raise Exception("')' expected at " + str(pos))
  else:
    raise Exception('invalid character at ' + str(pos))
  return e


def parse(s: str):
  global src, pos
  src, pos = s, 0
  nxt()
  e = expression()
  if sym != chr(0):
    raise Exception('unexpected character at ' + str(pos))
  return e
```

### Part B: Extended attribute grammar for counted repetitions

Extended grammar for `factor`:

```
factor(e) → atom(e) [ '*' « e := Star(e) »
                     | '+' « e := Conc(e, Star(e)) »
                     | '?' « e := Choice(e, ε()) »
                     | '{' integer(n)
                       ( '}' « e := repeat(e, n) »
                       | ',' ( integer(m) '}' « e := repeatRange(e, n, m) »
                             | '}' « e := Conc(repeat(e, n), Star(e)) »
                             )
                       )
                     ]
```

where:

- `repeat(e, 0) = ε()`, `repeat(e, n) = Conc(repeat(e, n-1), e)` for `n ≥ 1`
- `repeatRange(e, n, m) = Choice(repeat(e, n), repeatRange(e, n+1, m))` for `n < m`, `repeatRange(e, n, n) = repeat(e, n)`

Written with attribute rules:

```
factor(e) → atom(e) [ '*' « e := Star(e) »
    | '+' « e := Conc(e, Star(e)) »
    | '?' « e := Choice(e, ε()) »
    | '{' integer(n)
        ('}' « e := eN where e0 := ε(); ei := Conc(ei-1, e) for i = 1..n; eN := en »
        | ',' (integer(m) '}'
            « e := eN where e0 := ε(); ei := Conc(ei-1, e) for i = 1..m;
              eN := en | Choice(en, en+1) | ... | Choice(...Choice(en, en+1)..., em) »
            | '}' « eN := Conc(en, Star(e)) where e0 := ε(); ei := Conc(ei-1, e) for i = 1..n »
            )
        )
    ]
integer(n) → digit(n) { digit(d) « n := 10 × n + d » }
digit(d) → '0' « d := 0 » | ... | '9' « d := 9 »
```

_sol_:

```python
PlainChars = (
  ' !"#$%&\',-./0123456789:;<=>@ABCDEFGHIJKLMNO'
  + 'PQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz~'
)
EscapedChars = '()*+?\\|{}'
FirstFactor = PlainChars + '\\('

src: str
pos: int
sym: str


def nxt():
  global pos, sym
  if pos < len(src):
    sym, pos = src[pos], pos + 1
  else:
    sym = chr(0)


def repeat(e, n):
  r = ε()
  for _ in range(n):
    r = Conc(r, e)
  return r


def repeatRange(e, lo, hi):
  r = repeat(e, lo)
  for i in range(lo + 1, hi + 1):
    r = Choice(r, repeat(e, i))
  return r


def integer():
  if '0' <= sym <= '9':
    n = ord(sym) - ord('0')
    nxt()
  else:
    raise Exception('digit expected at ' + str(pos))
  while '0' <= sym <= '9':
    n = 10 * n + ord(sym) - ord('0')
    nxt()
  return n


def expression():
  e = term()
  while sym == '|':
    nxt()
    f = term()
    e = Choice(e, f)
  return e


def term():
  e = factor()
  while sym in FirstFactor:
    f = factor()
    e = Conc(e, f)
  return e


def factor():
  e = atom()
  if sym == '*':
    nxt()
    e = Star(e)
  elif sym == '+':
    nxt()
    e = Conc(e, Star(e))
  elif sym == '?':
    nxt()
    e = Choice(e, ε())
  elif sym == '{':
    nxt()
    n = integer()
    if sym == '}':
      nxt()
      e = repeat(e, n)
    elif sym == ',':
      nxt()
      if sym == '}':
        nxt()
        e = Conc(repeat(e, n), Star(e))
      else:
        m = integer()
        if sym == '}':
          nxt()
        else:
          raise Exception("'}' expected at " + str(pos))
        e = repeatRange(e, n, m)
    else:
      raise Exception("'}' or ',' expected at " + str(pos))
  return e


def atom():
  if sym in PlainChars:
    e = Sym(sym)
    nxt()
  elif sym == '\\':
    nxt()
    if sym in EscapedChars:
      e = Sym(sym)
      nxt()
    else:
      raise Exception('invalid escaped character at ' + str(pos))
  elif sym == '(':
    nxt()
    e = expression()
    if sym == ')':
      nxt()
    else:
      raise Exception("')' expected at " + str(pos))
  else:
    raise Exception('invalid character at ' + str(pos))
  return e


def parse(s: str):
  global src, pos
  src, pos = s, 0
  nxt()
  e = expression()
  if sym != chr(0):
    raise Exception('unexpected character at ' + str(pos))
  return e
```
