---
id: NFA
tags:
  - sfwr2fa3
date: "2024-01-30"
title: NFA
---

Ex. $\Sigma = {0, 1}$

1. $\mathcal{L}(M) = \{ abxba \mid x \in \Sigma^{*}\}$ ![[thoughts/university/sfwr-2fa3/abxba-nfa.png]]
1. $\mathcal{L}(M) = \{ yx \mid x = 00 \lor x =11 \land  y \in \Sigma^{*}\}$ ![[thoughts/university/sfwr-2fa3/yxx-nfa.png]]![[thoughts/university/sfwr-2fa3/yxx-nfa-4s.png]]

## $\epsilon$ transition
```mermaid
graph LR

  s((*)) --> s1{{s1}} --"1"--> s2{{s2}} --"1"--> s3{{s3}} --"e"--> s4{{s4}}
  s1{{s1}} --"e"--> s4{{s4}}
  s1{{s1}} --"0"--> s1{{s1}}
  s3{{s3}} --"1"--> s3{{s3}}
```
![[thoughts/university/sfwr-2fa3/eps-nfa.png]]

---
Given the following $M$
```mermaid
graph LR

  s((*)) --> s1{{s1}} --"1"--> s2{{s2}} --"1"--> s3{{s3}} --"e"--> s4{{s4}}
  s1{{s1}} --"e"--> s4{{s4}}
  s1{{s1}} --"0"--> s1{{s1}}
  s3{{s3}} --"1"--> s3{{s3}}
```

$\mathcal{L}(M) = \{0^n1^m \mid n \geq 0, m \neq 1 \space, x \in \Sigma^{*}\}$
