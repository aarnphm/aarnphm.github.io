---
date: "2024-12-08"
description: and exercise.
id: finals
modified: 2025-11-22 11:41:34 GMT-05:00
tags:
  - engineer4a03
title: Ethics template a la carte
---

```quotes
Tomorrow’s medicine is today’s research. That is why the question of how we allocate resources to research is at least as important as the question of how we allocate resources to health care itself.

Tony Hope
```

### Privacy and Confidentiality

1. Historical Context

   The Hippocratic Oath [@Edelstein1943EDETHO] emphasizes confidentiality as a sacred trust between physician and patient. Contemporary reiterations, such as the World Medical Association’s Declaration of Geneva, echo these sentiments, framing patient privacy as non-negotiable.

   The system must also implement Ann Cavoukian's Privacy by Design, putting emphasis on creating positive-sum than zero-sum. [@cavoukian2009privacy]

2. Philosophical consideration

   Enlightenment thinkers with the likes of Kant upheld human autonomy as fundamental [[thoughts/moral]] imperatives. [[thoughts/Philosophy and Kant|Kantian]] ethics suggests that
   using patient data solely as a means to an end, without their informed consent, is ethically problematic [@Kant1785KANGFT]. John Stuart [[thoughts/John Stuart Mill|Mill]]’s harm principle further
   supports protecting private information to prevent harm and maintain trust.

3. Contemporary implications

   In the age of data capitalism [@zuboff2019age], health data is a valuable commodity. Therefore, the AI system must strictly follow Ontario’s Personal Health Information Protection Act (PHIPA) and Canadian Federal Privacy Legislation (PIPEDA) [@phipaguide2004; @pipedaguide2024].
   Measures such as data minimization, differential privacy, encryption, and strict access controls must be in place. The framework should also include ongoing compliance checks and audits to ensure data-handling practices remain in line with evolving legal standards and community expectations.

4. Guidelines
   - Minimum necessary data collection principle
   - End-to-end encryption for all health data
   - Strict access controls and audit trails
   - Data localization within Canada to comply with PHIPA
   - Regular privacy impact assessments
   - Clear data retention and disposal policies

### Algorithmic [[thoughts/university/twenty-four-twenty-five/engineer-4a03/literature review#fairness|fairness]] and bias mitigation

1. Historical Context

   With the rise of predictive analytics and machine learning in the mid 2000s, the transition from pure statistical methods to more complex machine learning models began as computational power and large Medicare claims datasets became more accessible.
   ML models were developed to predict patient frailty, identify fraud and abuse, and forecast patient outcomes such as hospital readmission or mortality. [@obermeyer2016predicting; @raghupathi2014creating]

   However, researchers found that certain medicare data, reflecting decades of social inequality, could lead to predictive models that inadvertently disadvantaged some patients. For example, models predicting healthcare utilization might assign lower risk scores to communities with historically reduced access to care, not because they were healthier, but because they had fewer recorded encounters with the health system. [@obermeyer2019dissecting]

   Early mitigation attempts focused primarily on “fairness through awareness”—identifying and documenting biases. Health services researchers and policymakers began calling for the inclusion of demographic and social determinants of health data to correct for skewed historical patterns [@rajkomar2018ensuring].
   Some efforts were made to reweight training samples or stratify predictions by race, ethnicity, or income to detect differential performance.

2. Philosophical consideration

   John Rawls' _veil of ignorance_, or his principles of justice in general, encourage designing systems that benefit all segments of society fairly, without bias toward any particular group. [@rawls1999theory]
   Additionally, Nussbaum and Sen's capabilities approach suggests that technologies should expand human capabilities and [[thoughts/Agency]] (health, longevity, quality of life), especially marginalized communities. [@robeyns2020capability] [^emergent]

   Notable mentions that the AI system should also consider Kimberlé Crenshaw's theory of intersectionality in healthcare disparities to address fairness. [@crenshaw1991mapping]

[^emergent]:
    [[thoughts/LLMs|Large language model]] systems are poised to revolutionize ethnography by fundamentally altering how researchers conduct their work. In a sense, these systems should amplify our work, rather act as a replacement.
    Even these systems exhibit [[thoughts/emergent behaviour]] of [[thoughts/intelligence|intelligence]], we don't think it is artificial general intelligence ([[thoughts/AGI|AGI]]) due to [[thoughts/observer-expectancy effect]].

3. Contemporary implications

   Modern scholarship in data ethics [@noble2018algorithms] and public health frameworks stress the importance of addressing algorithmic bias. Recency bias in training data [@atlasofai] can disproportionately harm smaller rural communities, Indigenous populations, or minority groups
   who may not be well-represented in the data.

4. Guidelines
   - Rigorous bias audits of training datasets.
   - Engaging local communities (e.g., Northern Ontario Indigenous communities, diverse communities in Hamilton) in the development and testing phases.
   - Regularly updating and retraining models on more representative datasets.
   - Incorporating Kimberlé Crenshaw’s intersectionality framework to ensure that multiple axes of identity (e.g., Indigenous identity, rural location, age, disability) are considered.
   - Continual monitoring and transparent reporting on equity metrics over time.

### [[thoughts/mechanistic interpretability|Interpretability]] and transparency

1. Historical Context

   During the 1970s and 1980s, some of the earliest applications of AI were expert systems designed to replicate the decision-making abilities of human specialists—most notably in the medical domain [@10.7551/mitpress/4626.001.0001].
   One of the pioneering systems, MYCIN, developed at Stanford University in the 1970s, diagnosed and recommended treatments for blood infections [@shortliffe1974mycin].
   MYCIN’s developers recognized the importance of justifying recommendations, implementing what were known as “rule traces” to explain the system’s [[thoughts/reason|reasoning]]
   in human-understandable terms. Although these explanations were rudimentary, they established the principle that AI systems, especially those used in high-stakes domains
   like healthcare, should provide comprehensible justifications.

2. Philosophical consideration

   Hans-Georg Gadamer's work on hermeneutics highlight the importance of interpretation and understanding in human communication, including the relationship between patient, physician, and medical knowledge [@gadamer1977philosophical].
   Minimizing the opacity of AI models aligns with respecting patient autonomy and informed consent, as patients should understand how their health data influences recommendations.

3. Contemporary implications

   The AI systems must be equipped with user-friendly explanations of AI-driven recommendations. Rudin argued that for high-stakes decisions, it’s not merely desirable but often morally imperative to use interpretable models over post-hoc explanations of black boxes [@rudin2019stop].
   Thus, the AI system is required to be implemented with transparent algorithms. Additionally, Floridi suggests a unified principle where we must "[incorporate] both the epistemological sense of intelligibility (as an answer to the question 'how does it work?') and in the ethical sense of accountability (as an answer to the question: 'who is responsible for the way it works?')" in building the AI system. [@floridi2019unified]

   In Ontario and across Canada, healthcare data falls under stringent privacy and confidentiality laws. PHIPA in Ontario and the PIPEDA at the federal level mandate careful stewardship of personal health information.
   While these laws do not explicitly require explainable AI, their emphasis on accountability and trust indirectly encourages interpretable models into the AI systems.

   The emerging Artificial Intelligence and Data Act (AIDA), proposed under Bill C-27 at the federal level, signals Canada’s intention to regulate high-impact AI systems.
   The trajectory suggests future regulatory frameworks may explicitly demand that automated decision-making tools, such as our AI system—particularly in healthcare— must provide understandable rationales for their outputs [@aida2024companion].

4. Guidelines
   - Use of interpretable ML models where possible [@nanda2023concrete; @pozdniakov2024largelanguagemodelsmeet{pp. 3}; @cammarata2020thread]
   - Implements techniques like Local Interpretable Model-agnostic Explanations (LIME) [@ribeiro2016lime; @zafar2019dlime] or Shapley values
   - Clear communication of system limitations [@lundberg2017unified]
   - User-centric design principles, help ensure that clinicians, patients, and regulators can understand and trust the system.
   - Regular stakeholder feedback integration
   - Maintenance of model cards and datasheets [@mitchell2019model]