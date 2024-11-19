---
id: Exoplanets
tags:
  - astron2e03
date: "2024-02-02"
title: Expolanets
---

<!-- <div style="text-align: right"> -->
<!--   Aaron Pham - 400232791 -->
<!-- </div> -->

### Q1)

a. _Would you see any of the solar system planets transit?_

For an inclination of $i = 45 \degree$, transits are mostly observed when the orbital plan is edge on to the observer. It is plausible for some planets that is larger sized and orbit closer to ecliptic plane would transit the Sun given the direct line of sight.

b. _If you monitored the Sun with radial velocity (RV) measurements and your technology was precise enough that you could measure RV signals down to 1 m/s, show and discuss whether you’re able to detect Venus._

Given the semi-amplitude $K$ of the radial velocity curve is given by

$$
K = \frac{M_p \sin i}{(M_{*}+M_p)^{\frac{2}{3}}} \left( \frac{2 \pi G}{P} \right)^{\frac{1}{3}}
$$

We have

$$
\begin{align*}
  G &= 6.674 \times 10^{-11} m^3 \text{kg}^{-1} s^{-1} \\\
  M_p &= 4.87 \times 10^{24} \text{kg} \\\
  M_{*} &= 1.989 \times 10^{30} \text{kg} \\\
  P &= 224.7 \text { days} \\\
  K & = 4.87 \times 10^{24} \sin 45 \left( \frac{2 \pi G}{224.7 \times 24 \times 3600} \right)^{\frac{1}{3}}  \approx 0.061 \text{m/s}
\end{align*}
$$

Given the precision of the RV measurements is 1 m/s, we can conclude that Venus is not detectable with the current technology.

Venus induces a very small motion in the Sun due to gravitation pull, since RV is more sensitive to larger planets closer to their host stars.

c. _Using the same RV measurements, show and discuss whether you’re able to detect Jupiter_

For Jupiter, we have

$$
\begin{align*}
  G &= 6.674 \times 10^{-11} m^3 \text{kg}^{-1} s^{-1} \\\
  M_p &= 1.898 \times 10^{27} \text{kg} \\\
  M_{*} &= 1.989 \times 10^{30} \text{kg} \\\
  P &= 224.7 \text { days} \\\
  K = 1.898 \times 10^{27} \sin 45 \left( \frac{2 \pi G}{224.7 \times 24 \times 3600} \right)^{\frac{1}{3}}  \approx 8.81  \text{m/s}
\end{align*}
$$

We can conclude that Jupiter is detectable with the current technology.

This is due to Jupyter's significant mass and gravitational pull on the Sun, which induces a larger motion via the Doppler shifts.

d. _If you knew that the Sun’s mass is $1 M$ and you successfully detected Venus and/or Jupiter using these RV data, could you measure either planet’s absolute mass and why_

Detecting a planet using RV allows us to measure planet's minimum mass, not absolute mass. This has to do with the inclination angle of its orbit ($\sin i$)

If the orbit is edge-on ($i = 90 \degree$), then RV gives the closest approximation to the planet's absolute mass. However, in this case our $i = 45 \degree$, so we can only measure the minimum mass of the planet based on the assumption of an edge-on orbit.

e. _If you also monitored the Sun with astrometric measurements and your technology was precise enough that you could measure signals down to 10 $\mu \text{as}$ (i.e. micro-arcseconds), show and discuss whether you’re able to detect Jupiter_

The amplitude of astrometric signal $a$ is given by

$$
a = \frac{m_{p}}{m_{*}} \frac{a_{p}}{d}
$$

where $m_{p}$ is the mass of the planet, $m_{*}$ is the mass of the star, $a_{p}$ is the semi-major axis of the planet's orbit, and $d$ is the distance to the star.

For Jupyter, we have

$$
\begin{align*}
  m_{p} &= 1.898 \times 10^{27} \text{kg} \\\
  m_{*} &= 1.989 \times 10^{30} \text{kg} \\\
  a_{p} &= 5.2 \text{AU} \\\
  d &= 10 \text{pc} \\\
  a &= \frac{1.898 \times 10^{27}}{1.989 \times 10^{30}} \frac{5.2 \times 1.496 \times 10^{11}}{10 pc} * 1e^6 \approx 496.21 \mu \text{as}
\end{align*}
$$

Therefore, Jupyter would be easily detectable.

The signal is the result of Jupyter's substantial mass and larger distance from the Sun.

f. _Using the same astrometric measurements, show and discuss whether you’re able to detect Venus_

For Venus, we have

$$
\begin{align*}
  m_{p} &= 4.87 \times 10^{24} \text{kg} \\\
  m_{*} &= 1.989 \times 10^{30} \text{kg} \\\
  a_{p} &= 0.72 \text{AU} \\\
  d &= 10 \text{pc} \\\
  a &= \frac{4.87 \times 10^{24}}{1.989 \times 10^{30}} \frac{0.72 \times 1.496 \times 10^{11}}{10 pc} * 1e^6 \approx 0.177 \mu \text{as}
\end{align*}
$$

Therefore, Venus would not be detectable.

The signal is the result of Venus's smaller mass and closer proximity to the Sun, therefore exert a smaller gravitational effect on the Sun's position.

g. _If you knew that the Sun’s mass is 1 M and you successfully detected Venus and/or Jupiter using these astrometric data, could you measure either planet’s absolute mass and why?_

Yes, since astrometric measures the displacement of the star's position relative to distant background stars as it orbits around.

The amplitude of the astrometric signal is directly proportional to the mass of the planet, and inversely proportional to the mass of the star, therefore we can calculate the absolute mass of the planet, given the semi-major axis of its orbits and the mass of the stars (which is 1M in this case here).

### Q2)

$$
\begin{align*}
L_{\text{orb}} &= \frac{2 \pi a^2 \sqrt{1-e^2}}{P} M \\\
L_{\text{rot}} &= I \omega \\\
I &= \frac{2}{5} M R^2 \\\
\omega &= \frac{2 \pi}{P_{\text{rot}}}
\end{align*}
$$

a. _Derive the expression for the ratio of orbital to rotational angular momenta. For this exercise, assume a circular orbit_

For ratio $\frac{L_{\text{orb}}}{L_{\text{rot}}}$ we have

$$
\begin{align*}
L_{\text{orb}} &= \frac{2 \pi a^2}{P} M \\\
L_{\text{rot}} & = I \omega = \frac{2}{5} M R^2 \frac{2 \pi}{P_{\text{rot}}} = \frac{4 \pi M R^2}{5 P_{\text{rot}}}
\end{align*}
$$

Therefore $\frac{L_{\text{orb}}}{L_{\text{rot}}} = \frac{5 a^2 P_{\text{rot}}}{2 R^2 P}$

b. _It is a common misconception that the planets in our solar system orbit the Sun. In reality, the planets and the Sun all orbit their common center of mass. As such, the Sun has a non-zero semimajor axis $a_{\odot}$. Let us approximate the solar system as a 1-planet system that contains the Sun and Jupiter. In this scenario, what is the expression for $a_{\odot}$ in terms of Jupiter’s semimajor axis $a_J$ and both objects’ masses?_

In a two-body system, the formula to derive the distance of the Sun from the barycenter is given by:

$$
a_{\odot} = \frac{a_J M_J}{M_{\odot}}
$$

where $a_J$ is the semimajor axis of Jupiter, $M_J$ is the mass of Jupiter, and $M_{\odot}$ is the mass of the Sun.

The total distance $D$ between the Sun and Jupyter is the sum of their distance to the center of mass: $D = a_{\odot} + a_J$

Thus, considering this, the distance of the Sun from the barycenter is given by:

$$
a_{\odot} = \frac{a_J M_J}{M_J + M_{\odot}}
$$

c. _Using this expression, calculate the value of a in au_

Given that $a_J = 5.2 \text{AU}$, $M_J = 1.898 \times 10^{27} \text{kg}$, and $M_{\odot} = 1.989 \times 10^{30} \text{kg}$, we have

$$
a_{\odot} = \frac{5.2 \times 1.898 \times 10^{27}}{1.898 \times 10^{27} + 1.989 \times 10^{30}} \approx 0.00496 \text{AU}
$$

d. _Given your value of $a_\odot$, calculate the ratio of the Sun’s orbital angular momentum to its rotation angular momentum. Is most of the Sun’s angular momentum manifested as orbital or rotational?_

Using the formula derived in part a, we have

$$
\frac{L_{\text{orb}}}{L_{\text{rot}}} = \frac{5 a_{\odot}^2 P_{\text{rot}}}{2 R^2 P} = \frac{5 \times {0.00496 \text{AU}}^2 \times 25 * 86400 \text{ sec}}{2 \times {(6.96 \times 10^8)}^2 \times 11.86 \times 3.153 \times 10^7} \approx 0.0164
$$

This indicates that most of the Sun's angular momentum is manifested as rotational.

e. _Now calculate the ratio of Jupiter’s orbital angular momentum to its rotational angular momentum. Is most of Jupiter’s angular momentum manifested as orbital or rotational?_

Using the formula derived in part a, we have

$$
\frac{L_{\text{orb}}}{L_{\text{rot}}} = \frac{5 a_J^2 P_{\text{rot}}}{2 R^2 P} = \frac{5 \times {5.2 \text{AU}}^2 \times 9.93 \times 3600 \text{ sec}}{2 \times {(7.149 \times 10^7)}^2 \times 11.86 \times 3.153 \times 10^7} \approx 28287.8
$$

This indicates that most of Jupiter's angular momentum is manifested as orbital.

f. _In parts d) and e) above, you should have found that the total angular momenta of both the Sun and Jupiter are heavily dominated by either their own $Li_{\text{orb}}$ or $L_{\text{rot}}$. Using the dominant forms of angular momenta for each body, calculate the ratio $\frac{L_J}{L_\odot}$_

For Jupyter's orbital angular momentum $L_{\text{orb}, J}$, we have $L_{\text{orb}, J} = M_J \sqrt{G M_{\odot} a_J}$, and for the Sun's rotational angular momentum $L_{\text{rot}, \odot} = I_{\odot} \omega_{\odot}$, we have $L_{\text{rot}, \odot} = \frac{2}{5} M_{\odot} R_{\odot}^2 \omega_{\odot} = \frac{2}{5} M_{\odot} R_{\odot}^2 \frac{2 \pi}{P_{\text{rot,} \odot}}$

Thus the ratio $\frac{L_J}{L_\odot}$ is given by

$$
\frac{L_J}{L_\odot} = \frac{L_{\text{orb}, J}}{L_{\text{rot}, \odot}} = \frac{M_J \sqrt{G M_{\odot} a_J}}{\frac{2}{5} M_{\odot} R_{\odot}^2 \frac{2 \pi}{P_{\text{rot,} \odot}}}
$$

Given that $a_J = 5.2 \text{AU}$, $M_J = 1.898 \times 10^{27} \text{kg}$, $M_{\odot} = 1.989 \times 10^{30} \text{kg}$, $R_{\odot} = 6.96 \times 10^8 \text{m}$, and $P_{\text{rot,} \odot} = 25 \times 86400 \text{sec}$, we have

$$
\frac{L_J}{L_\odot} \approx 17.20
$$

g. _Comment on where most of the angular momentum in the solar system is located._

Most of angular momentum in the solar system is located in the orbital motion of the planets, with Jupyter having the most significant contribution to the total angular momentum.

This is due to the angular momentum of an orbiting body is proportional to the mass of the body and the distance from the center of mass, and inversely proportional to the period of the orbit.

### Q3)

$$
\begin{align}
v(\theta) &= \sqrt{GM \left( \frac{2}{r(\theta)} - \frac{1}{a} \right)} \\\
E = K + U &= -\frac{GMm}{2a} \\\
\end{align}
$$

a. _Use the conservation of angular momentum L and mechanical energy E to derive Eq. 4_

The angular momentum $L$ of a planet in orbit around a larger mass is given by

$$
L = mrv_{\perp}
$$

where:

- $m$ is the mass of the planet
- $v_{\perp}$ is the velocity of the planet perpendicular to the vector pointing from the Sun
- $r$ is the distance from the planet to the larger mass.

In an elliptical orbit, the direction of veloocity changes, but magnitude of angular momentum is conserved due to no external torques. Therefore

$$
L = mr(\theta)v(\theta)\sin \phi = \text{constant}
$$

The total mechanical energy $E$ of a planet in orbit around a larger mass is given by

The kinetic energy $K$ and the potential energy $U$ of a planet in orbit around a larger mass is given by

$$
\begin{align}
K &= \frac{1}{2}mv(\theta)^2 \\\
U &= -\frac{GMm}{r(\theta)}
\end{align}
$$

The total mechanical energy $E$ of a planet in orbit around a larger mass is given by

$$
E = K + U = = \frac{1}{2}mv(\theta)^2 - \frac{GMm}{r(\theta)}
$$

Given that the orbital velocity $v(\theta)$ is given by

$$
v(\theta) = \sqrt{GM \left( \frac{2}{r(\theta)} - \frac{1}{a} \right)}
$$

We can substitute $v(\theta)$ into the equation for $K$ to get

$$
K = GMm \left( \frac{1}{r(\theta)} - \frac{1}{2a} \right)
$$

Thus the total mechanical energy $E$ of a planet in orbit around a larger mass is given by

$$
\begin{align}
E = K + U &= GMm \left( \frac{1}{r(\theta)} - \frac{1}{2a} \right) - \frac{GMm}{r(\theta)} \\\
&= GMm \left( \frac{1}{r(\theta)} - \frac{1}{2a} - \frac{1}{r(\theta)} \right) \\\
&= -\frac{GMm}{2a}
\end{align}
$$

b. Use Eq. 4 to derive Eq. 3

$$
E = K + U = = \frac{1}{2}mv(\theta)^2 - \frac{GMm}{r(\theta)}
$$

Since E remains constant, given that the total energy in a bound orbit is negative, we have

$$
E = -\frac{GMm}{2a}
$$

where $a$ is the semi-major axis of the orbit.

We equate the two equations and solve for $v(\theta)$ to get

$$
\begin{align}
-\frac{GMm}{2a} &= \frac{1}{2}mv(\theta)^2 - \frac{GMm}{r(\theta)} \\\
v(\theta)^2 &= \frac{GM}{r(\theta)} \left( \frac{2}{r(\theta)} - \frac{1}{a} \right) \\\
v(\theta) &= \sqrt{GM \left( \frac{2}{r(\theta)} - \frac{1}{a} \right)}
\end{align}
$$
