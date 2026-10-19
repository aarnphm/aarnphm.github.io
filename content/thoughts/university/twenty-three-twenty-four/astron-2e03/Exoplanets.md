---
date: '2024-02-02'
description: exoplanet detection through radial velocity measurements calculating semi-amplitude for venus and jupiter with precision analysis.
id: Exoplanets
modified: 2026-06-05 15:08:34 GMT-04:00
tags:
  - astron2e03
title: Exoplanets
---

### Q1)

The setup uses an observer at $10\,\mathrm{pc}$ and an orbital inclination of $45^\circ$. The calculations below use circular orbits and the rounded input values in these coursework answers.

a. _Would you see any of the solar system planets transit?_

No. For the nearly coplanar solar-system orbits viewed at $i = 45^\circ$, each planet misses the Sun's projected disc. Here $i = 90^\circ$ means an edge-on orbit.

For a circular orbit, the smallest projected separation between the star and planet centres at conjunction is $a\lvert\cos i\rvert$. A transit requires

$$
a\lvert\cos i\rvert < R_\odot + R_p.
$$

Even Mercury, the closest planet, has $a\cos 45^\circ \approx 0.27\,\mathrm{au}$, while the Sun's radius is only $R_\odot \approx 0.00465\,\mathrm{au}$. The planets' actual orbital tilts and eccentricities cannot close that gap.

b. _If you monitored the Sun with radial velocity (RV) measurements and your technology was precise enough that you could measure RV signals down to 1 m/s, show and discuss whether you’re able to detect Venus._

For a circular orbit, the stellar RV semi-amplitude is

$$
K = \left(\frac{2\pi G}{P}\right)^{1/3}
    \frac{M_p\sin i}{(M_*+M_p)^{2/3}}.
$$

I use the exercise's rounded values and neglect orbital eccentricity:

$$
\begin{aligned}
G &= 6.674\times10^{-11}\,\mathrm{m^3\,kg^{-1}\,s^{-2}}, \\
M_* &= 1.989\times10^{30}\,\mathrm{kg}, \\
M_V &= 4.87\times10^{24}\,\mathrm{kg}, \\
P_V &= 224.7\times86400\,\mathrm{s}, \\
i &= 45^\circ.
\end{aligned}
$$

Substitution gives

$$
K_V = \left(\frac{2\pi(6.674\times10^{-11})}{224.7\times86400}\right)^{1/3}
\frac{(4.87\times10^{24})\sin45^\circ}
{(1.989\times10^{30}+4.87\times10^{24})^{2/3}}
\,\mathrm{m\,s^{-1}}
\approx 0.0606\,\mathrm{m\,s^{-1}}.
$$

Venus's signal is about sixteen times smaller than the stated detection threshold, so this instrument would not detect it under the exercise's threshold model. At fixed stellar mass and inclination, the formula gives $K\propto M_pP^{-1/3}$ when $M_p\ll M_*$. More massive planets and shorter periods produce larger RV signals.

c. _Using the same RV measurements, show and discuss whether you’re able to detect Jupiter_

Jupiter has $M_J = 1.898\times10^{27}\,\mathrm{kg}$ and $P_J = 11.86\,\mathrm{yr}$. Using a Julian year of $365.25$ days,

$$
K_J = \left(\frac{2\pi(6.674\times10^{-11})}{11.86\times365.25\times86400}\right)^{1/3}
\frac{(1.898\times10^{27})\sin45^\circ}
{(1.989\times10^{30}+1.898\times10^{27})^{2/3}}
\,\mathrm{m\,s^{-1}}
\approx 8.81\,\mathrm{m\,s^{-1}}.
$$

This is about nine times the threshold, so Jupiter is detectable in the stated model. Recovering its orbit also needs observations that cover enough of its roughly twelve-year period. A signal amplitude alone does not specify the observing duration or account for stellar activity.

d. _If you knew that the Sun’s mass is $1 M$ and you successfully detected Venus and/or Jupiter using these RV data, could you measure either planet’s absolute mass and why_

RV alone leaves a mass–inclination ambiguity. In the planet-mass limit, the inferred quantity is $M_p\sin i$, so an unknown inclination gives a minimum mass. An edge-on orbit has $\sin i = 1$ and reaches that minimum.

The assumed inclination matters here. If $i = 45^\circ$ is independently known, then the detected Jupiter's true mass can be recovered by dividing its measured minimum mass by $\sin45^\circ$, a factor of $\sqrt{2}$. If that angle is merely part of the simulated setup and is unknown to the observer, RV data alone cannot supply the true mass. Venus remains below the stated detection threshold.

e. _If you also monitored the Sun with astrometric measurements and your technology was precise enough that you could measure signals down to 10 $\mu \text{as}$ (i.e. micro-arcseconds), show and discuss whether you’re able to detect Jupiter_

Let $a_{\mathrm{rel}}$ be the semimajor axis of the planet's orbit relative to the Sun. The Sun's barycentric semimajor axis is

$$
a_* = \frac{M_p}{M_*+M_p}a_{\mathrm{rel}}.
$$

The angular semimajor axis $\alpha$ is $a_*/d$ in radians when both lengths use the same units. The definition of the parsec gives a convenient equivalent:

$$
\alpha\,[\mathrm{arcsec}] = \frac{a_*\,[\mathrm{au}]}{d\,[\mathrm{pc}]}.
$$

With Jupiter at $a_{\mathrm{rel}} = 5.2\,\mathrm{au}$ and the observer at $d = 10\,\mathrm{pc}$,

$$
\alpha_J = \frac{1.898\times10^{27}}{1.989\times10^{30}+1.898\times10^{27}}
\frac{5.2}{10}\times10^6\,\mu\mathrm{as}
\approx 496\,\mu\mathrm{as}.
$$

Jupiter's signal is about fifty times the astrometric threshold. For a circular orbit, inclination compresses the projected minor axis; the major axis retains this amplitude. The orbit still has to be separated from proper motion and annual parallax in a sufficiently long time series.

f. _Using the same astrometric measurements, show and discuss whether you’re able to detect Venus_

For Venus, use $a_{\mathrm{rel}} = 0.72\,\mathrm{au}$ at the same distance:

$$
\alpha_V = \frac{4.87\times10^{24}}{1.989\times10^{30}+4.87\times10^{24}}
\frac{0.72}{10}\times10^6\,\mu\mathrm{as}
\approx 0.176\,\mu\mathrm{as}.
$$

This is about fifty-seven times below the threshold, so Venus is undetectable in this model. Its small mass and small orbital radius both reduce the Sun's displacement. Astrometry measures that displacement; RV measures the line-of-sight component of its velocity.

g. _If you knew that the Sun’s mass is 1 M and you successfully detected Venus and/or Jupiter using these astrometric data, could you measure either planet’s absolute mass and why?_

Yes, for a detected planet with a well-measured astrometric orbit, distance, and stellar mass. The projected orbit constrains inclination as well as angular size. With distance, angular size gives $a_*$; the period and stellar mass then constrain the companion mass through Kepler's law and the barycentre relation.

This removes the inclination ambiguity of an RV-only mass estimate. Under the thresholds above, this applies to Jupiter. Venus's astrometric signal is too small to fit.

### Q2)

$$
\begin{aligned}
L_{\mathrm{orb}} &= \frac{2\pi M a^2\sqrt{1-e^2}}{P}, \\
L_{\mathrm{rot}} &= I\omega, \\
I &= \frac{2}{5}MR^2, \\
\omega &= \frac{2\pi}{P_{\mathrm{rot}}}.
\end{aligned}
$$

a. _Derive the expression for the ratio of orbital to rotational angular momenta. For this exercise, assume a circular orbit_

The supplied expressions use $M$ for the mass of the body, $a$ for its barycentric orbital semimajor axis, and $R$ for its radius. The moment of inertia assumes a uniform-density sphere. For a circular orbit,

$$
\begin{aligned}
L_{\mathrm{orb}} &= \frac{2\pi M a^2}{P}, \\
L_{\mathrm{rot}} &= I\omega
= \frac{2}{5}MR^2\frac{2\pi}{P_{\mathrm{rot}}}.
\end{aligned}
$$

Taking the ratio cancels the mass and the common angular factor:

$$
\boxed{\frac{L_{\mathrm{orb}}}{L_{\mathrm{rot}}}
= \frac{5}{2}\left(\frac{a}{R}\right)^2\frac{P_{\mathrm{rot}}}{P}.}
$$

Both length quantities and both period quantities must use matching units. Real stars and gas giants have nonuniform density, so the exercise's moment of inertia is an approximation.

b. _It is a common misconception that the planets in our solar system orbit the Sun. In reality, the planets and the Sun all orbit their common center of mass. As such, the Sun has a non-zero semimajor axis $a_{\odot}$. Let us approximate the solar system as a 1-planet system that contains the Sun and Jupiter. In this scenario, what is the expression for $a_{\odot}$ in terms of Jupiter’s semimajor axis $a_J$ and both objects’ masses?\_

The barycentre condition is

$$
M_\odot a_\odot = M_J a_{J,\mathrm{bar}}.
$$

If the question's $a_J$ means Jupiter's semimajor axis about the barycentre, the exact answer is therefore

$$
a_\odot = \frac{M_J}{M_\odot}a_J.
$$

A tabulated Sun–Jupiter separation instead describes the relative orbit. Writing that semimajor axis as $a_{\mathrm{rel}}$ avoids using one symbol for two distances:

$$
\begin{aligned}
a_{\mathrm{rel}} &= a_\odot+a_{J,\mathrm{bar}}, \\
a_\odot &= \frac{M_J}{M_\odot+M_J}a_{\mathrm{rel}}, \\
a_{J,\mathrm{bar}} &= \frac{M_\odot}{M_\odot+M_J}a_{\mathrm{rel}}.
\end{aligned}
$$

I use $a_{\mathrm{rel}} = 5.2\,\mathrm{au}$ for the numerical parts below. The two conventions differ by about one part in a thousand here.

c. _Using this expression, calculate the value of a in au_

Using the relative semimajor axis convention just defined,

$$
a_\odot = \frac{1.898\times10^{27}}{1.989\times10^{30}+1.898\times10^{27}}
(5.2\,\mathrm{au})
\approx 0.004957\,\mathrm{au}.
$$

With $1\,\mathrm{au} = 1.496\times10^{11}\,\mathrm{m}$ at the precision used here, this is $a_\odot\approx7.416\times10^8\,\mathrm{m}$. Jupiter's barycentric semimajor axis is correspondingly $a_{J,\mathrm{bar}}\approx5.1950\,\mathrm{au}$.

d. _Given your value of $a_\odot$, calculate the ratio of the Sun’s orbital angular momentum to its rotation angular momentum. Is most of the Sun’s angular momentum manifested as orbital or rotational?\_

Use $P_{\mathrm{rot},\odot}=25\,\mathrm{d}$, $R_\odot=6.96\times10^8\,\mathrm{m}$, and the shared orbital period $P=11.86\times365.25\,\mathrm{d}$. Converting the orbital radius to metres first gives

$$
\frac{L_{\mathrm{orb},\odot}}{L_{\mathrm{rot},\odot}}
= \frac{5}{2}
\left(\frac{7.416\times10^8}{6.96\times10^8}\right)^2
\frac{25}{11.86\times365.25}
\approx 0.0164.
$$

In this model, the Sun's rotational angular momentum is about sixty-one times its barycentric orbital angular momentum.

e. _Now calculate the ratio of Jupiter’s orbital angular momentum to its rotational angular momentum. Is most of Jupiter’s angular momentum manifested as orbital or rotational?_

Use Jupiter's barycentric semimajor axis from part c, $R_J=7.149\times10^7\,\mathrm{m}$, and $P_{\mathrm{rot},J}=9.93\,\mathrm{h}$:

$$
\frac{L_{\mathrm{orb},J}}{L_{\mathrm{rot},J}}
= \frac{5}{2}
\left(\frac{5.1950\times1.496\times10^{11}}{7.149\times10^7}\right)^2
\frac{9.93}{11.86\times365.25\times24}
\approx 2.82\times10^4.
$$

Jupiter's orbital angular momentum is about twenty-eight thousand times its rotational angular momentum under the uniform-sphere approximation.

f. _In parts d) and e) above, you should have found that the total angular momenta of both the Sun and Jupiter are heavily dominated by either their own $Li_{\text{orb}}$ or $L_{\text{rot}}$. Using the dominant forms of angular momenta for each body, calculate the ratio $\frac{L_J}{L_\odot}$\_

The dominant terms are Jupiter's orbital angular momentum and the Sun's rotational angular momentum. Using the same orbital radius and period as above,

$$
\begin{aligned}
L_{\mathrm{orb},J}
&= \frac{2\pi M_J a_{J,\mathrm{bar}}^2}{P}
\approx 1.925\times10^{43}\,\mathrm{kg\,m^2\,s^{-1}}, \\
L_{\mathrm{rot},\odot}
&= \frac{2}{5}M_\odot R_\odot^2
\frac{2\pi}{P_{\mathrm{rot},\odot}}
\approx 1.121\times10^{42}\,\mathrm{kg\,m^2\,s^{-1}}.
\end{aligned}
$$

Thus, with the requested dominant-term approximation,

$$
\frac{L_J}{L_\odot}
\approx \frac{L_{\mathrm{orb},J}}{L_{\mathrm{rot},\odot}}
\approx 17.2.
$$

g. _Comment on where most of the angular momentum in the solar system is located._

The two-body calculation places most of its angular momentum in Jupiter's orbital motion, even though the Sun contains most of the mass. In the full solar system, planetary orbital motion also dominates, with Jupiter providing the largest planetary contribution.

For a nearly circular planet orbit with $M_p\ll M_\odot$, Kepler's law gives

$$
L_{\mathrm{orb},p}\approx M_p\sqrt{GM_\odot a_p}.
$$

This identifies the relevant dependence: orbital angular momentum grows with planet mass and with the square root of orbital size. The calculation above uses the exercise's uniform-sphere moments of inertia, so its numerical ratio should not be treated as a measured solar-system angular momentum budget.

### Q3)

The two expressions to derive are

$$
\begin{aligned}
v(\theta) &= \sqrt{GM\left(\frac{2}{r(\theta)}-\frac{1}{a}\right)}
&&\text{(Eq. 3)}, \\
E &= -\frac{GMm}{2a}
&&\text{(Eq. 4)}.
\end{aligned}
$$

a. _Use the conservation of angular momentum L and mechanical energy E to derive Eq. 4_

Take a bound Keplerian ellipse around a dominant central mass $M$, with an orbiting mass $m\ll M$. Gravity acts along the separation vector, so its torque about the central mass vanishes and $L$ is conserved. Mechanical energy is also conserved:

$$
L = mrv_\perp,
\qquad
E = \frac{1}{2}mv^2-\frac{GMm}{r}.
$$

At periapsis and apoapsis, the radial velocity is zero. Write their radii as $r_p=a(1-e)$ and $r_a=a(1+e)$. The velocity is perpendicular to the radius at each turning point, so $v_p=L/(mr_p)$ and $v_a=L/(mr_a)$. Equating their energies gives

$$
\frac{L^2}{2m r_p^2}-\frac{GMm}{r_p}
= \frac{L^2}{2m r_a^2}-\frac{GMm}{r_a}.
$$

For a noncircular ellipse, factor and cancel $1/r_p-1/r_a$:

$$
\frac{L^2}{2m}\left(\frac{1}{r_p}+\frac{1}{r_a}\right)=GMm,
\qquad
L^2 = \frac{2GMm^2r_pr_a}{r_p+r_a}.
$$

Substitute this into the energy at periapsis:

$$
\begin{aligned}
E &= \frac{GMm\,r_a}{r_p(r_p+r_a)}-\frac{GMm}{r_p} \\
&= -\frac{GMm}{r_p+r_a}
= \boxed{-\frac{GMm}{2a}}.
\end{aligned}
$$

This obtains Eq. 4 without assuming the velocity formula in Eq. 3. For a circular orbit, $r_p=r_a=a$ and that cancellation is unavailable. Instead, radial force balance gives $v^2=GM/a$; substitution into the energy gives the same result.

b. Use Eq. 4 to derive Eq. 3

Substitute Eq. 4 into the conserved energy expression and solve for the speed:

$$
\begin{aligned}
-\frac{GMm}{2a}
&= \frac{1}{2}mv(\theta)^2-\frac{GMm}{r(\theta)}, \\
v(\theta)^2
&= GM\left(\frac{2}{r(\theta)}-\frac{1}{a}\right), \\
v(\theta)
&= \sqrt{GM\left(\frac{2}{r(\theta)}-\frac{1}{a}\right)}.
\end{aligned}
$$

This is the vis-viva equation. Its squared-speed units are correct because $GM$ has units $\mathrm{m^3\,s^{-2}}$ and the bracket has units $\mathrm{m^{-1}}$. At $r=a$, it reduces to the circular-orbit speed $\sqrt{GM/a}$.

### Sources and assumptions

These revised answers retain the questions from my submitted [[thoughts/university/twenty-three-twenty-four/astron-2e03/Aaron_Pham_400232791_phama10_assignment1.pdf|Assignment 1]]. The submission records the original work; the corrections above do not alter that PDF. The course's [[thoughts/university/twenty-three-twenty-four/astron-2e03/Solar System Exoplanets 2024.pdf|Solar System and Exoplanets slides]] cover the RV minimum-mass ambiguity and astrometric reflex motion (PDF pages 53, 59, and 77).

[NASA's detection-method overview](https://science.nasa.gov/exoplanets/how-we-find-and-characterize/) describes RV and transit observations. The [NASA Exoplanet Archive's calculation notes](https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html) give the RV semi-amplitude and astrometric amplitude conventions. [JPL's planetary physical parameters](https://ssd.jpl.nasa.gov/planets/phys_par.html) provide reference masses and periods. The arithmetic here keeps the exercise's rounded inputs rather than mixing them with higher-precision catalog values.
