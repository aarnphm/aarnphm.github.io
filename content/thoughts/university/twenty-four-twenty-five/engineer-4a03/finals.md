---
id: finals
tags:
  - engineer4a03
date: "2024-12-08"
description: and exercise.
modified: 2024-12-08 17:47:01 GMT-05:00
title: Scenarios for development.
---

## [[thoughts/ethics|ethics]] consideration

<p class="quotes">
  Tomorrow’s medicine is today’s research. That is why the question of how we allocate resources to research is at least as important as the question of how we allocate resources to health care itself. -- Tony Hope, <i>Medical Ethics</i>
</p>

### Privacy and Confidentiality

1. Historical Context

The Hippocratic Oath [@Edelstein1943EDETHO] emphasizes confidentiality as a sacred trust between physician and patient. Contemporary reiterations, such as the World Medical Association’s Declaration of Geneva, echo these sentiments, framing patient privacy as non-negotiable.

The system must also implement Ann Cavoukian's Privacy by Design, putting emphasis on creating positive-sum than zero-sum. [@cavoukian2009privacy]

2. Philosophical consideration

Enlightenment thinkers with the likes of Kant upheld human autonomy as fundamental [[thoughts/moral]] imperatives. [[thoughts/Philosophy and Kant|Kantian]] ethics suggests that
using patient data solely as a means to an end, without their informed consent, is ethically problematic [@Kant1785KANGFT]. John Stuart [[thoughts/university/twenty-three-twenty-four/philo-1aa3/John Stuart Mill|Mill]]’s harm principle further
supports protecting private information to prevent harm and maintain trust.

3. Contemporary considerations

In the age of data capitalism [@zuboff2019age], health data is a valuable commodity. Therefore, the AI system must strictly follow Ontario’s Personal Health Information Protection Act (PHIPA) and Canadian Federal Privacy Legislation (PIPEDA) [@phipaguide2004; @pipedaguide2024].
Measures such as data minimization, differential privacy, encryption, and strict access controls must be in place. The framework should also include ongoing compliance checks and audits to ensure data-handling practices remain in line with evolving legal standards and community expectations.

4. Guidelines

- Minimum necessary data collection principle
- End-to-end encryption for all health data
- Strict access controls and audit trails
- Data localization within Canada to comply with PHIPA
- Regular privacy impact assessments
- Clear data retention and disposal policies

### Algorithmic Fairness and Bias mitigation

1. Historical Context

[^ref]
