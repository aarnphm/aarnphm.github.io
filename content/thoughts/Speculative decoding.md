---
id: Speculative decoding
tags:
  - ml
  - serving
  - technical
date: "2025-05-21"
description: a method to speed up LLM decoding
modified: 2025-05-21 09:59:07 GMT-04:00
title: Speculative decoding
---

Idea: "draft-and-verify" using smaller models to generate a head tokens (quick explanation from [karpathy](https://x.com/karpathy/status/1697318534555336961))

Intuitively:

- we generate a small set of lookahead tokens, albeit 2-5 tokens with smaller speculators
- uses the larger models to "verify" the input sequences + draft tokens (then replace tokens that aren't valid from rejection sampler)

In a sense, we are verify these in parallel instead of [[thoughts/Autoregressive models|autoregressive decoding]].

A few techniques such as [[thoughts/Speculative decoding#ngrams|ngrams]], [[thoughts/Speculative decoding#EAGLE|EAGLE]] are supported in [[thoughts/vllm|vLLM]]

## MLP Speculator

_via combined tokens/embedding speculators_

abs: https://arxiv.org/abs/2404.19124v1

## SPiRE

## MagicDec

## ngram

## EAGLE
