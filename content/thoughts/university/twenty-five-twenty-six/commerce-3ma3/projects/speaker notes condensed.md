---
date: "2026-03-26"
id: speaker notes condensed
modified: 2026-03-26 11:28:10 GMT-04:00
tags:
  - commerce3ma3
title: Speaker Notes (Condensed)
---

## 1. Title (0:15)

- Group 5, Mister Maki research readout

## 2. TOC (0:15)

- 7 sections, ~14 min

## 3. Main Question (0:30)

- 81% awareness, 49% trial, 32pp gap = the whole story

## 4. Background (0:50)

- Only temaki in Hamilton, 990 King St W, Spring 2025
- MDP: allocate limited marketing budget
- MRP: awareness, familiarity, value, trial, catering

## 5. Exploratory Research (0:40)

- Owner consult: 80/20 student split, seasonal decline, 60% repeat, 4.8/5 TGTG, catering 10%
- Secondary: Pita Pit ~$10.50, Bandidos ~$10-12, Saigon ~$8-14, AYCE ~$27-32. MM at $12-18 tops quick-service
- Digital: 1,342 IG followers vs 37,592 students. Zero r/McMaster. Zero Silhouette.

## 6. Objectives (0:30)

- RO1-5 funnel. Primary survey covers RO1-4. RO5 via owner consult only.

## 7. Methodology (0:40)

- 1 week March 2026, Qualtrics, n=37 (25 students, 12 residents), 30 questions
- Python (scipy, statsmodels). Convenience + snowball. 15% of target.

## 8. Key Findings (0:40)

- Product excellent. Funnel is the problem.

## 9. Awareness (0:40)

- 54% unaided (highest), 81% aided, 89% WOM

## 10. Conversion Gap (0:50) [PIVOT]

- 81% → 49%. Students 44%, residents 58.3%. H3 confirms students price-sensitive.

## 11. Value Perception (0:45)

- Non-customers 2x "poor value" (21% vs 11%). 22% customers "excellent", 0% non-customers. 4.11/5 portion/price.

## 12. Satisfaction (0:40)

- 4.83/4.78/4.72/4.56/4.11. Reorder 4.44. 94% walk-in. Acquisition > retention.

## 13. Stats Detail (0:25)

- 6 key numbers

## 14. Hypotheses List (0:25)

- 10 hypotheses, 5 test types, alpha 0.05

## 15. Hypothesis Testing (1:30)

- **H3 REJECT**: value × segment, U=73.5, p=0.005, r=0.51. Residents BETTER value. FLIPS hypothesis.
- **H5 REJECT**: satisfaction dims, chi2=13.83, p=0.008. Portion/price outlier.
- **H8 REJECT**: reorder × satisfaction, rho=0.544, p=0.020
- **H4 BORDERLINE**: r=0.323, p=0.051. ~5 more respondents would flip.
- 6 not significant. n=37 detects only large effects.

## 16. Statistical Detail (0:30)

- Familiarity: 0% → 17% → 57% → 60% → 57%. Dining: occasional/regular 63-67%.

## 17. Recs 1-2 (0:50)

- Educate on temaki (22% low familiarity)
- Student combo deal (H3: students price-sensitive)

## 18. Recs 3-4 (0:40)

- Referral: bring friend, both get roll (89% WOM)
- Pilot catering: 20-50 tasters/event

## 19. Limitations (0:40)

- n=37 vs 381 ideal (9.7%). Convenience sampling. Self-report. Cross-sectional. No competitor primary data. RO5 no primary data.

## 20. Quote (0:10)

- [Pause]

## 21. Summary (0:40)

- Education + trial incentives, not awareness. Product not the problem, funnel is.

## 22. Thank You

- Q&A

## Appendix A: Formal H0/H1 (not presented)

## Appendix B: Study Scope & Power (not presented)

---

## Q&A PREP

| Topic                   | Answer                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Ideal sample size       | 381 (Cochran's, 95% CI, 5% margin, pop ~37,000)                                          |
| Software                | Python (scipy.stats, statsmodels, pandas)                                                |
| Why Fisher's exact      | Expected cells < 5 at n=37                                                               |
| Why Mann-Whitney        | Ordinal Likert, non-normal                                                               |
| Competitive positioning | **NOT in survey.** RO3 via secondary price comparison only.                              |
| Catering data (RO5)     | **NOT in survey.** Owner consultation only. 10% revenue, format ideal for events.        |
| H3 reversal             | Original predicted residents lower value. Data shows opposite. Students price-sensitive. |
| Bonferroni              | Corrected alpha = 0.005. H3 (p=0.005) STILL significant. H5/H8 would not survive.        |
| RO coverage             | RO1,2,4 full. RO3 partial. RO5 secondary only.                                           |
