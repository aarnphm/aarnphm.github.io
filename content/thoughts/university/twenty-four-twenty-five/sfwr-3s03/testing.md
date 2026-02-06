---
date: '2025-03-03'
description: and summary
id: testing
modified: 2025-10-29 02:16:05 GMT-04:00
tags:
  - sfwr3s03
title: Software Testing with LLM Survey
---

## problem context

Software testing is crucial for ensuring quality and reliability, but as software systems grow more complex, traditional testing approaches face limitations. Meanwhile, Large Language Models (LLMs) have emerged as powerful tools for various tasks, including code generation. This paper explores the intersection of LLMs and software testing to understand how these models can address testing challenges.

## paper's contribution

This comprehensive survey analyzes 102 research papers that use LLMs for software testing.
The analysis is conducted from both software testing and LLM perspectives, identifying:

- Common testing tasks where LLMs are employed (test case preparation, debugging, program repair)
- Which LLMs are most utilized (ChatGPT, Codex, CodeT5, etc.)
- Prompt engineering approaches (zero-shot, few-shot learning, chain-of-thought)
- Challenges and opportunities in this emerging field

## interested audience

- Software engineering researchers, particularly those focused on testing and AI-assisted development
- Testing practitioners looking to integrate LLMs into their workflows
- AI researchers interested in domain-specific applications of LLMs
- Software development organizations seeking to improve testing efficiency

## software testing techniques explored

- Unit test case generation and test oracle generation
- System test input generation
- Fuzz testing for detecting vulnerabilities
- Metamorphic testing for test oracle problems
- Differential testing for finding inconsistencies
- Mutation testing for test quality evaluation
- Bug analysis, localization and repair

## software systems focused on

- Mobile applications
- Deep learning libraries
- Compilers
- SMT solvers
- Autonomous driving systems
- Cyber-physical systems
- JavaScript engines
- Video games

## key challenge

Achieving high testing coverage remains difficult with LLMs. Despite using high temperature settings to generate diverse outputs,
LLMs still struggle to adequately explore the vast space of possible behaviors.

## open challenges and future work

- Exploring LLMs in early testing stages (requirements analysis, test planning)
- Applying LLMs to integration and acceptance testing, which are currently underexplored
- Investigating LLMs for non-functional testing (performance, security, usability)
- Developing advanced prompt engineering techniques beyond zero-shot/few-shot approaches
- Creating better combinations of LLMs with traditional testing techniques
- Building specialized benchmarks that haven't been included in LLM training data
- Addressing real-world deployment challenges in enterprise environments
- Fine-tuning specialized LLMs specifically for testing tasks
