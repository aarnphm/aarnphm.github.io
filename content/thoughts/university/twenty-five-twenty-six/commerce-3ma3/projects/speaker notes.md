---
date: '2026-03-26'
id: speaker_notes
modified: 2026-03-26 11:28:10 GMT-04:00
tags:
  - commerce3ma3
title: Mister Maki Research Readout — Speaker Notes
---

## Group 5 | Commerce 3MA3 | Target: 14 minutes + 1 min buffer

---

### Slide 1: Title (0:15)

"Good [morning/afternoon]. We're Group 5. Today we're presenting our marketing research findings for Mister Maki, Hamilton's only dedicated temaki restaurant, located in Westdale."

---

### Slide 2: Table of Contents (0:15)

"Quick roadmap: we'll cover the research question, background, methodology, four key findings, our statistical hypothesis testing, recommendations, and limitations. About 14 minutes."

---

### Slide 3: Main Question (0:30)

"The management question: how can Mister Maki convert high awareness into actual trial? 81% of respondents recognize the brand. Only 49% have ordered. That 32 percentage point gap is the entire story, and everything in this presentation flows from it."

---

### Slide 4: Background (0:50)

"Mister Maki opened in spring 2025 at 990 King Street West, directly adjacent to McMaster's campus. They offer grab-and-go temaki, hand rolls priced at $4 to $7 each, with a typical meal running $12 to $18 for 2 to 3 rolls.

The management decision problem: how should they allocate limited marketing budget to increase customer acquisition?

The marketing research problem: determine awareness levels, assess whether the temaki concept itself creates a familiarity barrier, evaluate perceived value relative to competitors like Pita Pit and Burrito Bandidos, identify trial drivers, and gauge catering demand among student organizations."

---

### Slide 5: Exploratory Research (0:40)

"Before designing the survey, we conducted three streams of exploratory research.

First, an owner consultation. We learned that roughly 80% of customers are students, the business experiences severe seasonal decline during breaks, the repeat rate is about 60%, and catering represents about 10% of revenue.

Second, secondary research. We compared menu pricing across Westdale competitors: Pita Pit at about $10.50, Burrito Bandidos $10 to $12, Saigon Asian $8 to $14. Mister Maki's $12 to $18 for a meal is at the top of this range, approaching all-you-can-eat sushi pricing.

Third, we assessed digital presence. Mister Maki has about 1,300 Instagram followers versus 37,000 enrolled McMaster students. Zero mentions on r/McMaster. No coverage in The Silhouette. The brand has near-zero digital footprint despite strong in-person awareness."

---

### Slide 6: Research Objectives (0:30)

"Five research objectives, structured as a funnel. RO1 and RO2 ask whether people know and understand the product. RO3 evaluates perceived value. RO4 identifies what drives purchase and repeat. RO5 explores catering as a growth channel. Note: our primary survey directly addresses RO1 through RO4. RO5 is informed by the owner consultation."

---

### Slide 7: Methodology (0:40)

"One-week data collection in March 2026, via Qualtrics. 37 respondents: 25 McMaster students and 12 Hamilton residents. 30 questions spanning awareness, satisfaction, and value perception. Analysis conducted in Python using scipy and statsmodels.

Convenience plus snowball sampling. Important to note: our target was 250 respondents and we achieved 37, about 15% of target. All findings are directional, and our statistical power is limited to detecting large effects."

---

### Slide 8: Key Findings Overview (0:40)

"The headline: the product is excellent, the funnel is the problem. Mister Maki has built strong organic awareness without heavy marketing spend. But awareness without understanding and perceived value does not convert to trial.

Four findings on the right: the 32 percentage point awareness-to-trial gap, word-of-mouth as the dominant channel, near-ceiling satisfaction scores with portion-price as the one weak spot, and a value perception gap between customers and non-customers."

---

### Slide 9: Brand Awareness Stats (0:40)

"Three numbers. 54% unaided recall, highest of any Westdale restaurant. 81% aided awareness. And 89% of restaurant discovery happens through word-of-mouth.

The brand awareness engine works. The question is why it isn't converting."

---

### Slide 10: Conversion Gap — Finding 1 (0:50)

"This is the pivot slide. 81% awareness to 49% trial. A 32 percentage point gap.

Students convert at 44%, residents at 58.3%. Our hypothesis testing confirmed why: students perceive significantly worse value than residents. We'll get to that in the statistical analysis section.

Nearly half the people who know Mister Maki have never ordered. That is the single largest growth opportunity."

---

### Slide 11: Value Perception — Finding 2 (0:45)

"Non-customers are twice as likely to rate the price as poor value: 21% versus 11%.

The irony: among people who've actually tried it, 22% rate it excellent value. Zero non-customers do. The product sells itself once tried, but the price tag creates a barrier to that first purchase.

Portion size versus price scored 4.11 out of 5, the weakest satisfaction dimension and the only one where a customer reported dissatisfaction."

---

### Slide 12: Customer Satisfaction — Finding 3 (0:40)

"Among the 18 customers: near-ceiling satisfaction. Speed 4.83, quality 4.78, packaging 4.72.

Reorder intent 4.44 out of 5. 61% will definitely reorder. 94% order via walk-in takeout, delivery is essentially unused.

The takeaway: retention is not the problem. Acquisition is."

---

### Slide 13: Stats Detail (0:25)

"Six framing numbers: 68% students in sample, 44% versus 58% trial rates by segment, 94% walk-in, 76% have tried temaki somewhere but only 49% from Mister Maki, and 22% have low concept familiarity."

---

### Slide 14: Hypotheses List (0:25)

"We tested 10 hypotheses across six research areas. Tests used include Fisher's exact for small-sample categorical comparisons, Mann-Whitney U for ordinal between-group comparisons, Friedman for repeated measures, and Spearman and point-biserial correlations. All at alpha 0.05."

---

### Slide 15: Hypothesis Testing Results (1:30)

"Three hypotheses rejected at the 0.05 level.

H3: value perception differs significantly by segment. Mann-Whitney U 73.5, p 0.005, with a large effect size r 0.51. This finding is surprising: residents perceive BETTER value than students. Median rating 4 versus 3. This flips our original hypothesis, which predicted residents would perceive lower value. Students are the price-sensitive segment, which explains their lower 44% conversion rate.

H5: satisfaction varies significantly across the five dimensions. Friedman chi-square 13.83, p 0.008. Portion size versus price at 4.11 is statistically lower than speed of service at 4.83.

H8: reorder intent positively correlates with mean satisfaction. Spearman rho 0.544, p 0.020. Among our 18 customers, higher satisfaction predicts stronger reorder intent.

H4, temaki familiarity predicting trial, came in at p 0.051. Borderline. With approximately 5 more respondents, this would likely reach conventional significance.

Six hypotheses were not significant. With n equals 37, we can only reliably detect large effects, Cohen's d above 0.92. Non-significant results should not be interpreted as evidence of no effect."

---

### Slide 16: Statistical Detail (0:30)

"Two dose-response patterns. Trial rate by familiarity climbs from 0% at level 1 to 57-60% at levels 3 through 5. Trial by dining frequency: occasional and regular diners convert at 63-67%, while never and rarely diners show 0%.

Neither reached significance at n=37, but both show patterns consistent with a familiarity-drives-trial mechanism. The satisfaction bars at the bottom confirm the Friedman result: portion-to-price in pink is the outlier."

---

### Slide 17: Recommendations 1-2 (0:50)

"Four recommendations, each grounded in specific findings.

First: educate on the temaki concept. 22% have low familiarity. The format itself is the barrier. In-store sampling, social media showing what temaki is and how to eat it, visual menu guides.

Second: a student combo deal. H3 confirmed students are the price-sensitive segment. A combo with a drink or side reframes price-to-portion without discounting the core product."

---

### Slide 18: Recommendations 3-4 (0:40)

"Third: a referral program. 89% discover via word-of-mouth. The engine exists, give it a mechanism. Bring a friend, both get a free roll.

Fourth: pilot catering for McMaster clubs. Each event puts 20 to 50 hand rolls in front of first-time tasters. Currently 10% of revenue. The temaki format is ideal: individually wrapped, no plating. Catering doubles as a sampling tool."

---

### Slide 19: Limitations (0:40)

"Transparency about what this study can and cannot tell you. Small sample, 37 versus target 250, 15% achievement. Convenience sampling over-represents McMaster-connected respondents. Self-reported satisfaction may contain social desirability bias. Cross-sectional design, single point in time. No direct competitor primary data.

With more time: larger sample, longitudinal component, intercept surveys at competitor locations."

---

### Slide 20: Big Quote (0:10)

[Pause. Let it land.]

---

### Slide 21: Summary (0:40)

"Here's what we recommend: invest in education and trial incentives, not awareness. The brand is already known. The product already delivers.

Three pillars: educate consumers on temaki through sampling and visual guides. Reframe value with a student combo deal. Leverage the 89% word-of-mouth engine with a referral program and catering pilot.

The product is not the problem. The funnel is."

---

### Slide 22: Thank You (Q&A)

"Thank you. We welcome your questions."

---

### Appendix A: Formal Hypotheses (not presented, for Q&A reference)

### Appendix B: Study Scope & Power Analysis (not presented, for Q&A reference)

---

**Total speaking time: ~14:30** (with 0:30 buffer to 15:00)

## TIMING SUMMARY

| Section                           | Slides | Time                                  |
| --------------------------------- | ------ | ------------------------------------- |
| Opening + Setup                   | 1-4    | 1:45                                  |
| Exploratory + Objectives + Method | 5-7    | 1:50                                  |
| Findings                          | 8-13   | 3:40                                  |
| Statistical Analysis              | 14-16  | 2:25                                  |
| Recommendations                   | 17-18  | 1:30                                  |
| Close                             | 19-22  | 1:30                                  |
| Appendix                          | A-B    | 0:00 (not presented)                  |
| **Total**                         | **21** | **~12:05 speaking + pauses = ~13:50** |

## DECK CRITIQUE & GAPS

Before the notes, a few things to be aware of during Q&A and for the final report:

**Strengths of this deck:**

- Clear narrative arc: problem → method → findings → stats → recs → limits
- Every recommendation tied to a specific data point
- H3 finding (value × segment) is genuinely surprising and publishable-quality
- Honest about limitations and statistical power
- Research Readout template is clean, professional, legible at projection size

**Gaps to prepare for in Q&A:**

- **Competitive positioning**: the survey did NOT include rating questions for MM vs competitors. RO3 is addressed via secondary price comparison only (menu prices from websites). If asked, be upfront: "We compared pricing from public menu data. The survey instrument focused on awareness, trial, and value perception rather than head-to-head competitive ratings."
- **Catering/RO5**: catering interest, club involvement, and event budget questions were NOT in the final survey. RO5 is informed by the owner consultation only (~10% of revenue from catering, format is ideal for events). If asked: "RO5 was scoped based on our owner consultation. The survey prioritized RO1-4 given our sample constraints."
- **Ideal sample size**: Cochran's formula, 95% CI, 5% margin, population ~37,000 gives n=381. We achieved 37 (9.7% of ideal). This is an honest limitation.
- **Analysis software**: Python (scipy.stats, statsmodels, pandas). The report requires naming this explicitly.
- **H0/H1 format**: the report requires proper null/alternative formulations. Formal versions are in the appendix slides and the report. e.g., "H0: there is no significant difference in value perception between students and residents. H1: there is a significant difference."
