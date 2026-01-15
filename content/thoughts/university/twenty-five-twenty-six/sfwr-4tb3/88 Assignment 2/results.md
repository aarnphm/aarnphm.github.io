---
date: "2026-01-15"
description: AST et al.
id: results
modified: 2026-01-15 12:47:09 GMT-05:00
tags:
  - sfwr4tb3
  - assignment
title: regular constructions and language parser
---

## A1

Consider expression made up of identifiers $a, b, c, d$ and binary operators $+, -$ like shown

$$
\begin{align*}
a &+ b &+ c \\
a &- b &- c \\
a &- b &+ c - d \\
a &+ b &- c + d
\end{align*}
$$

Write grammars as below with NLTK and draw the parse trees with NLTK

> [!question] 1.
>
> Write a grammar such that $+$ binds tighter than $-$ and both $+$ and $-$ associate to the left. That is, $a+b+c$ is parsed as $\left( a+b \right)+c$ and $a-b+c-d$ as $\left( a- \left( b+c \right) \right) - d$. Draw the parse tree for $a+b+c$ and $a-b+c-d$

> [!question] 2.
> Write a grammar such that $-$ binds tighter than $+$ and both $-$ and $+$ associate to the left. That is $a+b+c$ is parsed as

## A2

## A3

## A4
