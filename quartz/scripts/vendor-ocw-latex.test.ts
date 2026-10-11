// Failure list for the LaTeX converter (docs/plans/course-resources.md §4). Each case is a way the
// conversion was seen to go wrong on the 18.156 source; E2E coverage is the vendored notes themselves.
import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  convertLatexDocument,
  expandMacros,
  parseAux,
  parseSynctex,
  parseToc,
  pageOfLine,
  scanPreamble,
  parseLatex,
  splitRows,
  stripComments,
} from './vendor-ocw-latex'

const preambleSrc = String.raw`
\documentclass{amsart}
\usepackage{amsmath}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{Definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\newtheorem{ques}{Question}
\DeclareMathOperator{\Hopf}{Hopf}
\newcommand{\T}{\mathbb{T}}
\newcommand{\TT}{\mathcal{T}}
\newcommand{\eps}{\epsilon}
`

function convert(body: string, mode: 'notes' | 'pset' = 'notes') {
  return convertLatexDocument(`${preambleSrc}\n\\begin{document}\n${body}\n\\end{document}\n`, {
    mode,
    noteSlug: n => `courses/x/notes/${String(n).padStart(2, '0')}`,
    pageLink: () => null,
  })
}

describe('vendor-ocw-latex', () => {
  it('expands macros on control-word boundaries', () => {
    const preamble = scanPreamble(parseLatex(stripComments(preambleSrc)))
    assert.equal(
      expandMacros(String.raw`\T \TT \Theta \Hopf(S)`, preamble),
      String.raw`\mathbb{T} \mathcal{T} \Theta \operatorname{Hopf}(S)`,
    )
  })

  it('keeps only the number in a statement header when the title carries math', () => {
    const doc = convert(String.raw`
\section{one}
\begin{lemma}[Elementary $L^2$ bounds on $f$]
Body text here.
\end{lemma}
`)
    const md = doc.sections[0].markdown
    assert.match(md, /^> \[!lemma\|latex\] 1\.1$/m)
    assert.match(md, /^> _\(Elementary \$L\^2\$ bounds on \$f\$\)_$/m)
    assert.match(md, /^\^lemma-1-1$/m)
  })

  it('does not count a row break inside substack as an equation', () => {
    const doc = convert(String.raw`
\section{one}
\begin{align}
a &= \sum_{\substack{i \\ j}} b \label{first} \\
c &= d \label{second}
\end{align}
`)
    assert.equal(doc.labels.get('first')?.value, '1')
    assert.equal(doc.labels.get('second')?.value, '2')
    assert.deepEqual(splitRows(String.raw`x \\ \begin{cases} a \\ b \end{cases} \\ y`).length, 3)
  })

  it('turns a forced break in prose into a paragraph break and strips SETUP bold', () => {
    const doc = convert(String.raw`
\section{one}
\textbf{SETUP}\\
Let $f$ be a function.
`)
    const md = doc.sections[0].markdown
    assert.match(md, /^\*\*SETUP\*\*$/m)
    assert.match(md, /^Let \$f\$ be a function\.$/m)
  })

  it('escapes markdown-active text at a line start', () => {
    const doc = convert(String.raw`
\section{one}
\# not a heading here.

1. not a list because of the escape.
`)
    const md = doc.sections[0].markdown
    assert.match(md, /^\\# not a heading/m)
    assert.match(md, /^\\1\. not a list/m)
  })

  it('does not let a callout title swallow the body', () => {
    const doc = convert(String.raw`
\section{one}
\begin{theorem}(Double Counting Real Version)\label{dc}\\
For every $x$, there is $y$.
\end{theorem}
`)
    const md = doc.sections[0].markdown
    assert.match(md, /^> \[!theorem\|latex\] 1\.1 Double Counting Real Version$/m)
    assert.match(md, /^> For every \$x\$, there is \$y\$\.$/m)
    assert.equal(doc.labels.get('dc')?.anchor, 'theorem-1-1')
  })

  it('keeps an unknown environment as a fenced block and reports it', () => {
    const doc = convert(String.raw`
\section{one}
\begin{weird}[x]
content
\end{weird}
`)
    const md = doc.sections[0].markdown
    assert.match(md, /^```latex\n\\begin\{weird\}\[x\]\ncontent\n\\end\{weird\}\n```$/m)
    assert.ok(doc.sections[0].unhandled.some(entry => entry.startsWith('environment weird')))
  })

  it('numbers subfigures by letter and questions on their own counter', () => {
    const doc = convert(String.raw`
\section{one}
\begin{figure}
\centering
\begin{subfigure}{0.4\textwidth}\caption{left}\label{fa}\end{subfigure}
\begin{subfigure}{0.4\textwidth}\caption{right}\label{fb}\end{subfigure}
\caption{both}\label{fig}
\end{figure}
\begin{ques}First\end{ques}
\section{two}
\begin{ques}Second\end{ques}
\begin{theorem}Reset\end{theorem}
`)
    assert.equal(doc.labels.get('fa')?.value, '1a')
    assert.equal(doc.labels.get('fb')?.value, '1b')
    assert.equal(doc.labels.get('fig')?.value, '1')
    assert.match(doc.sections[0].markdown, /\[!question\|latex\] Question 1$/m)
    assert.match(doc.sections[1].markdown, /\[!question\|latex\] Question 2$/m)
    assert.match(doc.sections[1].markdown, /\[!theorem\|latex\] 2\.1$/m)
  })

  it('renders item labels and enumerate targets', () => {
    const doc = convert(String.raw`
\section{one}
\begin{enumerate}
\item[(i)] first \label{it1}
\item second
\end{enumerate}
So \ref{it1} holds.
`)
    const md = doc.sections[0].markdown
    assert.match(md, /^1\. \*\*\(i\)\*\* first$/m)
    assert.match(md, /^2\. second$/m)
    assert.match(md, /So 1 holds\./)
  })

  it('reports duplicate labels and renders unknown references as ??', () => {
    const doc = convert(String.raw`
\section{one}
\begin{lemma}\label{dup}A\end{lemma}
\begin{lemma}\label{dup}B\end{lemma}
See \ref{dup} and \ref{missing}.
`)
    assert.ok(doc.report.some(entry => entry.startsWith('duplicate label dup')))
    assert.ok(doc.report.some(entry => entry.startsWith('undefined reference missing')))
    assert.match(
      doc.sections[0].markdown,
      /See \[\[courses\/x\/notes\/01#\^lemma-1-2\|1\.2\]\] and \?\?\./,
    )
  })

  it('links cross-section references, equations and proofs', () => {
    const doc = convert(String.raw`
\section{one}
\begin{theorem}\label{t}X\end{theorem}
\begin{proof}[Proof sketch]Y\end{proof}
\begin{equation}a=b\label{e}\end{equation}
\section{two}
By theorem \ref{t} and \eqref{e}.
`)
    const one = doc.sections[0].markdown
    assert.match(one, /^> \[!proof\|latex\]\+ Proof sketch of theorem 1\.1$/m)
    assert.match(one, /^\^proof-1-1$/m)
    assert.match(one, /^<a id="eq-1" class="latex-eq"><\/a>$/m)
    assert.match(one, /\\tag\{1\}/)
    assert.match(
      doc.sections[1].markdown,
      /\[\[courses\/x\/notes\/01#\^theorem-1-1\|1\.1\]\] and \[\[courses\/x\/notes\/01#eq-1\|\(1\)\]\]/,
    )
  })

  it('reads a date line and leaves it out of the body', () => {
    const doc = convert(String.raw`
\section{one}
Thursday Feb $20$.

Content.
`)
    assert.equal(doc.sections[0].date, '02-20')
    assert.doesNotMatch(doc.sections[0].markdown, /Feb/)
  })

  it('flags KaTeX failures instead of emitting them', () => {
    const doc = convert(String.raw`
\section{one}
$\undefinedmacro x$
`)
    assert.equal(doc.katexErrors.length, 1)
  })

  it('parses aux, toc and synctex', () => {
    const aux = parseAux(
      String.raw`\newlabel{a}{{1.1}{3}}\newlabel{sub@b}{{a}{4}}\newlabel{c}{{2}{5}}`,
    )
    assert.equal(aux.get('a')?.value, '1.1')
    assert.equal(aux.has('sub@b'), false)
    const toc = parseToc(String.raw`\contentsline {section}{\tocsection {}{1}{Title}}{3}
\contentsline {section}{\tocsection {}{2}{$\mathbb {R}$ things}}{8}`)
    assert.equal(toc.get(2), 8)
    const pages = parseSynctex(
      'SyncTeX Version:1\nInput:1:main.tex\n\nContent:\n{1\n[1,10:0,0\n(1,12:1\n}1\n{2\n[2,5:0\n[1,40:0\n}2\n',
    )
    assert.deepEqual(pages, [
      { page: 1, firstLine: 10 },
      { page: 2, firstLine: 40 },
    ])
    assert.equal(pageOfLine(pages, 39), 1)
    assert.equal(pageOfLine(pages, 40), 2)
    assert.equal(pageOfLine(pages, 5), null)
  })
})
