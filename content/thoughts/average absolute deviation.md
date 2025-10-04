---
id: average absolute deviation
tags:
  - seed
  - math
description: Summary statistic of variability
source: https://en.wikipedia.org/wiki/Average_absolute_deviation
date: "2025-09-18"
created: "2025-09-18"
modified: 2025-10-04 04:29:42 GMT-04:00
published: "2003-10-18"
title: average absolute deviation
---

The average absolute deviation (AAD) of a data set is the average of the absolute deviations from a central point. It is a summary statistic of statistical dispersion or variability. In the general form, the central point can be a mean, median, mode, or any other measure of central tendency or reference value. AAD includes the mean absolute deviation and the median absolute deviation (both abbreviated as MAD).

## Measures of dispersion

Several measures of statistical dispersion are defined in terms of the absolute deviation. The term "average absolute deviation" does not uniquely identify a measure, as there are several measures that can be used to measure absolute deviations, and there are several measures of central tendency that can be used as well.

> Thus to uniquely identify the absolute deviation it is necessary to specify ==both the measure of deviation and the measure of central tendency==.
>
> The statistical literature has not yet adopted a standard notation, as both the mean absolute deviation around the mean and the median absolute deviation around the median have been denoted by "MAD" in the literature, which may lead to confusion, since they generally have values considerably different from each other.

## Mean absolute deviation around a central point

For arbitrary differences (not around a central point), see Mean absolute difference. For paired differences (also known as mean absolute deviation), see Mean absolute error.

For a set $X = \{x_1, x_2, \ldots, x_n\}$, the mean absolute deviation around a central point $m(X)$ is

$$
\mathrm{AAD}(X; m) = \frac{1}{n} \sum_{i=1}^n \lvert x_i - m(X) \rvert.
$$

The choice of $m(X)$ affects the value. For the data set $\{2,2,3,4,14\}$:

- Arithmetic mean $= 5$:
  $$
  \frac{|2-5|+|2-5|+|3-5|+|4-5|+|14-5|}{5} = 3.6.
  $$
- Median $= 3$:
  $$
  \frac{|2-3|+|2-3|+|3-3|+|4-3|+|14-3|}{5} = 2.8.
  $$
- Mode $= 2$:
  $$
  \frac{|2-2|+|2-2|+|3-2|+|4-2|+|14-2|}{5} = 3.0.
  $$

### Mean absolute deviation around the mean

The mean absolute deviation (MAD), also referred to as the \\"mean deviation\\" or sometimes \\"average absolute deviation\
