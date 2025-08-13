---
id: MoE
tags:
  - seed
description: Mixture of Expert
date: "2025-08-13"
modified: 2025-08-13 14:45:31 GMT-04:00
title: MoE
---

## Step3

[@step3system], stored in bfloat16 or block-fp8, used for vision-language reasoning.

proposes [[thoughts/Attention#MFA]] to reduces KVCache and attention cost -- 22% of DeepSeek V3's per-token attention cost.
