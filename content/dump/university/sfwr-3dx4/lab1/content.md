---
id: content
tags:
  - sfwr-3dx4
date: "2024-01-24"
title: PID Controller
---
### prelab.

### lab.

#### 5.1
$$
\begin{align}
\frac{\theta}{V} &= \frac{K}{s((Js+b)(Ls+R) + K^{2})}  \\\
& = \frac{K}{s(JLs^{2}+bLs + JRs+bR + K^{2})} \\\
G(s) & = \frac{K}{JLs^{3} + s^{2}(bL+JR) + (K^2+bR)s} \\\
\end{align}
$$
![[dump/university/sfwr-3dx4/lab1/5.1-graph.png]]
1. What does the graph represents? What does the first derivative of the graph represent and look like?
	- Angular position of the motor. The first derivate would be the angular velocity, or the rate of change. It would start at zero (as the first part is flatten) then will keep increasing since the slope is positive.
2. What is represented by non-linear section?
	- Represent the system is accelerating
3. Steady-state error
4. percent overshoot
5. settling time of this response
6. is the response stable with respect to angular position?

#### 5.2
![[dump/university/sfwr-3dx4/lab1/5.2-graph.png]]