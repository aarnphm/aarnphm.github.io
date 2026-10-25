---
date: '2024-09-04'
description: how pagerank scores pages from links alone, from the 1998 stanford report through the random surfer, dangling pages, convergence and personalization, and how this garden ranks its notes with it.
id: PageRank
modified: 2026-10-25 11:30:00 GMT-04:00
seealso:
  - '[[thoughts/Search|Search]]'
  - '[[pagerank]]'
tags:
  - technical
  - algorithm
title: PageRank
---

PageRank gives every page one score computed from the link graph alone. Larry Page, Sergey Brin, Rajeev Motwani and Terry Winograd describe it in a Stanford report ([Page et al., 1999](http://ilpubs.stanford.edu:8090/422/)); the draft is dated January 29, 1998. The score does not depend on the query, so a [[thoughts/Search|search engine]] computes it once per crawl and uses it for every query. This garden computes the same score over its own notes on [[pagerank|/pagerank]].

## From citations to links

Citation analysis counts how often other papers cite a paper. The report treats a link as a citation, so a page with many backlinks (inbound links) is probably important. Raw backlink counts fail on the web for two reasons. A web page costs nothing to publish, so anyone can make many pages that link to one target. And one link from the Yahoo home page says more about a page than many links from pages that nobody reads.

PageRank uses a recursive rule instead: a page has a high rank when the ranks of the pages that link to it add up to a large number. Many backlinks raise a page, and so do a few backlinks from high-ranked pages. A page with no rank of its own passes on almost nothing, so a farm of new pages adds little.

## The simplified rank

Let $B_u$ be the set of pages that link to page $u$, and let $N_v$ be the number of links on page $v$. The report first defines

$$
R(u) = c \sum_{v \in B_u} \frac{R(v)}{N_v}
$$

Each page divides its rank equally among its links, and each page receives the sum of the parts that arrive. The constant $c$ keeps the total rank fixed.

The report's Figure 3 is a small example. A links to B and C, B links to C, and C links to A. The ranks $R(A) = 0.4$, $R(B) = 0.2$, $R(C) = 0.4$ are a fixed point: A sends $0.2$ on each of its two links, C receives $0.2$ from A and $0.2$ from B, and C sends all of its $0.4$ to A.

In matrix form, let $M_{uv} = 1/N_v$ when $v$ links to $u$, and $0$ otherwise. Then $R = cMR$, so $R$ is an eigenvector of $M$.[^transpose] The report wants the dominant eigenvector, which power iteration finds: start from almost any vector and multiply by $M$ until the result stops changing.

## Rank sinks and the random surfer

The simplified rule fails on closed loops. Take two pages that link only to each other, and a third page that links to one of them. Rank flows into the pair on every iteration and never leaves. The report calls this a rank sink.

The fix is a rank source $E$, a vector over pages:

$$
R'(u) = c \sum_{v \in B_u} \frac{R'(v)}{N_v} + c\,E(u), \qquad \lVert R' \rVert_1 = 1
$$

The report explains $E$ with a random surfer. The surfer starts on a random page and keeps clicking links at random. A real person caught in a small loop would get bored and go to some other page. $E$ is the distribution of those jumps. Most of the report's experiments used a uniform $E$ with $\lVert E \rVert_1 = 0.15$.

The form used today writes the same model with a damping factor $d = 0.85$. At each step the surfer follows a random link on the current page with probability $d$ and jumps to a uniformly random page with probability $1 - d$. With $N$ pages and $L(q)$ links on page $q$:

$$
\mathrm{PR}(p) = \frac{1-d}{N} + d \sum_{q \to p} \frac{\mathrm{PR}(q)}{L(q)}
$$

$\mathrm{PR}(p)$ is the long-run fraction of steps that the surfer spends on $p$.[^anatomy] Without the jump, the surfer in a rank sink stays there forever. With the jump, the surfer leaves any loop within about $1/(1-d) \approx 6.7$ steps on average.

## Dangling pages

A page with no links is dangling. Its rank flows nowhere, so the total rank shrinks on every iteration. In 1998 most dangling pages were pages that the crawler had not fetched yet: the 24 million downloaded pages pointed to 51 million URLs that were not downloaded. The report removes dangling pages, computes the ranks of the others, adds the dangling pages back, and iterates a few more times.

The usual choice now, and the one this garden makes, keeps dangling pages and treats each one as a link to every page. The surfer on a dangling page jumps at random:

$$
\mathrm{PR}(p) = \frac{1-d}{N} + \frac{d}{N}\sum_{L(q)=0}\mathrm{PR}(q) + d \sum_{q \to p} \frac{\mathrm{PR}(q)}{L(q)}
$$

With this term the scores always sum to $1$, and no rank is lost.

## Computing it

The report's algorithm keeps the jump term separate from the link matrix:

$$
\begin{aligned}
R_{i+1} &\leftarrow A R_i \\
\delta_{\text{lost}} &\leftarrow \lVert R_i \rVert_1 - \lVert R_{i+1} \rVert_1 \\
R_{i+1} &\leftarrow R_{i+1} + \delta_{\text{lost}}\, E \\
\delta &\leftarrow \lVert R_{i+1} - R_i \rVert_1
\end{aligned}
$$

It repeats until $\delta < \epsilon$. Each step follows the links once, measures the rank that leaked out, and adds that amount back spread over $E$. (The report names the leaked amount $d$; it is a different quantity from the damping factor.)

The 1998 implementation gave each URL an integer ID and sorted the link table by source page. Only the current weights stayed in memory: 4 bytes for each of 75 million URLs is 300 MB. The sorted link table and the previous weights streamed from disk. One iteration took about 6 minutes on a workstation, and the whole computation took about 5 hours. The report notes that this cost is small next to building the full-text index.

On 322 million links the computation reached its tolerance in about 52 iterations. On half the links it took about 45. Twice the data cost about 15% more iterations. The report explains the fast convergence with expander graphs: a random walk mixes in time logarithmic in the size of the graph when the gap between the first and second eigenvalues is large.

A sharper result came later. For any link graph, the second eigenvalue of the PageRank matrix satisfies $\lvert\lambda_2\rvert \le d$ ([Haveliwala and Kamvar, 2003](http://ilpubs.stanford.edu:8090/582/)). The error of power iteration therefore shrinks by at least a factor of $d$ on every step, for any shape of web. The jump term also makes the Markov chain irreducible and aperiodic, so by the Perron–Frobenius theorem the stationary distribution exists and is unique. With $d = 0.85$, the change between steps falls below $10^{-10}$ after at most

$$
k = \frac{\ln\left(2 \cdot 10^{10}\right)}{\ln(1/0.85)} \approx 146
$$

iterations, because the first change is at most $2$ in $L_1$. A smaller $d$ converges faster but gives the links less weight against the uniform jump.

## Personalized PageRank

$E$ decides which pages get rank without any links. A uniform $E$ values every page simply because it exists. The report found one problem with it: groups of pages with many links among themselves, such as copyright notices, disclaimers and heavily interlinked mailing-list archives, get too much rank.

The other extreme puts all of $E$ on one page. The report tried the Netscape home page and the home page of John McCarthy. In both cases that page ranked first, followed by the pages that it links to. Under McCarthy's $E$, pages about computer science and Stanford ranked higher. The home page of John Mitchell, in the Stanford theory group, was at the 100.00 percentile under McCarthy's view and at the 93.89 percentile under Netscape's. A search for "Mitchell" with McCarthy's ranks returns that colleague first among the many people named Mitchell.

Personalization also resists spam. To raise a page's rank, an author must persuade an important page, or many pages, to link to it. When $E$ sits on one trusted page, the author must win links from pages close to that page. The report suggests a middle ground, $E$ over the root pages of all web servers, and notes its weakness: anyone can create many servers that all point at one site.

## Search and other uses

To test PageRank for search, the authors built a search engine over the titles of 16 million pages. It finds every page whose title contains all the query words and sorts them by PageRank. A query for "university" returned a list of major universities, while AltaVista returned pages that looked random. PageRank helps most with underspecified queries: for "Stanford University" it puts the university home page first, above the many pages that mention Stanford. Google combines PageRank with text-match scores, word proximity and anchor text (the text of links that point to a page).

The report lists other uses:

1. **traffic estimation.** Pages with heavy traffic but low PageRank are pages that people visit but do not link to.
2. **crawl order.** On the Stanford web, PageRank computed on a partial crawl predicts a page's final backlink count better than the backlink count seen so far. Backlink counting stays inside the local cluster it started in, while PageRank finds the Stanford home page early and moves out from it.
3. **navigation.** The PageRank proxy marks every link on a page with a bar whose length is the log of the target's PageRank, so a reader can see which links in a long list matter before clicking. On a Stanford people page, one professor's link had a very low rank, which showed that its URL was wrong.

## In this garden

[[pagerank|/pagerank]] runs the dangling-page formula above over the notes of this garden:

- The nodes are the public notes. A link counts once for each pair of notes, a note's links to itself are dropped, and links to hidden notes are dropped.
- $d = 0.85$. The computation starts from $1/N$ on every note and stops when the $L_1$ change is at most $10^{-10}$, or after 160 iterations. 160 is above the bound of 146, so the cap never stops a run early.
- The table shows $x_p = N \cdot \mathrm{PR}(p)$, so $1.00\times$ is the mean note.

In October 2026 the graph had 2,095 notes and 2,925 links, and the computation converged in 113 iterations. The top 10% of notes held 53% of the score. 1,243 notes, or 59%, had no inbound link. Each of those notes gets only the jump and dangling terms, about $0.38\times$ the mean, which is also the median score.

The five highest notes were all course home pages, with MIT 18.156 at about $76\times$ the mean. This is the mailing-list problem from the report. A course home page links to its syllabus, assignments and lectures, and those pages link back, so rank circulates inside the course. The 18.156 syllabus has two inbound links and ranked 12th. The page has a courses toggle that hides course navigation pages, and homepage suggestions skip them.

The quick-open palette with an empty query lists notes near the current one with personalized PageRank. $E$ is the current note, links count in both directions, and the surfer always jumps back to the current note. This is the McCarthy experiment with one note as the home page. Search results also show each note's global rank as a small mark.

[^transpose]: The report writes $A_{u,v} = 1/N_u$ for a link from $u$ to $v$ and then $R = cAR$. With those indices, $(AR)(u)$ is the average rank of the pages that $u$ links to. The equation needs the transpose, $R = cA^\top R$, which is the matrix $M$ here.

[^anatomy]: The companion paper on Google ([Brin and Page, 1998](<https://doi.org/10.1016/S0169-7552(98)00110-X>)) writes $\mathrm{PR}(A) = (1-d) + d\sum_i \mathrm{PR}(T_i)/C(T_i)$ and says that the scores form a probability distribution that sums to one. Without the $1/N$ on the first term, the scores sum to $N$. The order of the pages does not change, because the missing $1/N$ multiplies every score by the same factor. The same paragraph calls $d$ the probability that the surfer gets bored, which reverses the meaning of $d = 0.85$.
