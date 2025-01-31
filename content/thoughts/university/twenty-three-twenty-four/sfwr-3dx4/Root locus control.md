---
id: Root locus control
tags:
  - sfwr3dx4
date: "2024-02-28"
modified: 2024-12-17 17:45:48 GMT-05:00
title: Root locus control
---

See also [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/root_locus_control.pdf|slides]] and [[thoughts/Root locus]]

closed-loop properties of the function of $K_1 G(s)$

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 652.1217638634798 291.00981025621786" width="652.1217638634798" height="291.00981025621786">
  <!-- svg-source:excalidraw -->

  <defs>
    <style class="style-fonts">
      @font-face {
        font-family: "Virgil";
        src: url("https://excalidraw.com/Virgil.woff2");
      }
      @font-face {
        font-family: "Cascadia";
        src: url("https://excalidraw.com/Cascadia.woff2");
      }
      @font-face {
        font-family: "Assistant";
        src: url("https://excalidraw.com/Assistant-Regular.woff2");
      }
    </style>

  </defs>
  <g stroke-linecap="round" transform="translate(102.69564715984916 12.754350387616796) rotate(0 28.59265624498471 28.592656244984482)"><path d="M28.23 -1.14 C33.55 -1.99, 40.43 -0.16, 45.21 4.13 C49.98 8.43, 55.2 18.12, 56.9 24.63 C58.61 31.14, 58.21 38.34, 55.44 43.2 C52.68 48.07, 45.99 51.66, 40.32 53.83 C34.66 56.01, 27.58 57.69, 21.46 56.25 C15.35 54.81, 6.91 50.42, 3.62 45.18 C0.33 39.95, 0.81 30.93, 1.74 24.85 C2.68 18.77, 1.23 11.19, 9.23 8.7 C17.24 6.2, 41.99 6.39, 49.76 9.88 C57.53 13.38, 56.54 29.88, 55.86 29.66 M15.4 4.02 C20.78 0.45, 29.24 -2.54, 35.32 -1.54 C41.39 -0.53, 47.87 4.96, 51.84 10.06 C55.8 15.15, 59.58 22.78, 59.1 29.03 C58.63 35.27, 53.14 42.85, 48.97 47.55 C44.8 52.25, 39.91 56.4, 34.07 57.22 C28.23 58.04, 19.8 55.64, 13.93 52.49 C8.06 49.34, 0.64 44.25, -1.13 38.31 C-2.9 32.36, 0.36 22.94, 3.31 16.83 C6.26 10.71, 15.22 3.61, 16.59 1.61 C17.96 -0.39, 11.52 4.51, 11.52 4.83" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(126.57424771399837 31.12894550963665) rotate(0 4.495994567871094 10)"><text x="4.495994567871094" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="middle" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">x</text></g><g stroke-linecap="round" transform="translate(241.87104576286038 15.279605468890622) rotate(0 39.84175645754749 29.584534787217308)"><path d="M14.79 0 C26.69 0.94, 41.29 1.11, 64.89 0 M14.79 0 C34.03 -0.77, 50.34 -1.63, 64.89 0 M64.89 0 C75.16 -1.16, 78.15 4.32, 79.68 14.79 M64.89 0 C73.68 -1.18, 80.02 4.25, 79.68 14.79 M79.68 14.79 C78.38 24.96, 83.52 25.5, 79.68 44.38 M79.68 14.79 C81.75 23.8, 81.36 31.28, 79.68 44.38 M79.68 44.38 C80.71 52.09, 73.45 58.34, 64.89 59.17 M79.68 44.38 C82.34 54.92, 74.69 58.08, 64.89 59.17 M64.89 59.17 C49.87 60.27, 31.21 63.06, 14.79 59.17 M64.89 59.17 C55.86 60.21, 41.2 58.67, 14.79 59.17 M14.79 59.17 C1.64 56.39, 0.47 58.06, 0 44.38 M14.79 59.17 C7.12 58.75, 0.39 53.63, 0 44.38 M0 44.38 C-2.73 38.22, -1.77 24.71, 0 14.79 M0 44.38 C-0.53 32.37, 1.74 21.8, 0 14.79 M0 14.79 C3.39 3.43, 1.1 -1.42, 14.79 0 M0 14.79 C0.48 4.09, 4.93 0, 14.79 0" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(276.8088028917946 34.86414025610793) rotate(0 4.903999328613281 10)"><text x="4.903999328613281" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="middle" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">K</text></g><g stroke-linecap="round" transform="translate(417.08730142273816 23.115706288102956) rotate(0 39.84175645754749 29.584534787217308)"><path d="M14.79 0 C36.31 -1.4, 54.71 -0.88, 64.89 0 M14.79 0 C24.66 -0.11, 35.25 -0.59, 64.89 0 M64.89 0 C78.04 -2.85, 83.2 1.72, 79.68 14.79 M64.89 0 C74.33 2.76, 79.67 9.17, 79.68 14.79 M79.68 14.79 C81.55 23.56, 81.67 40.45, 79.68 44.38 M79.68 14.79 C78.12 23.73, 81.16 34.81, 79.68 44.38 M79.68 44.38 C81.71 57.09, 72.07 56.61, 64.89 59.17 M79.68 44.38 C79.01 51.86, 75.93 57.49, 64.89 59.17 M64.89 59.17 C55.06 60.25, 40.89 57.09, 14.79 59.17 M64.89 59.17 C48.05 58.53, 35.07 58.67, 14.79 59.17 M14.79 59.17 C6.6 61.96, 1.48 52.55, 0 44.38 M14.79 59.17 C5.86 63.2, -0.62 52.35, 0 44.38 M0 44.38 C-4.09 29.62, -3.89 23.22, 0 14.79 M0 44.38 C-1.62 34.12, 1.18 21.71, 0 14.79 M0 14.79 C2.42 1, 8.08 1.46, 14.79 0 M0 14.79 C-4.45 7.1, 8.43 0.38, 14.79 0" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(440.24907283389894 42.700241075320264) rotate(0 16.67998504638672 10)"><text x="16.67998504638672" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="middle" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">G(s)</text></g><g stroke-linecap="round"><g transform="translate(166.03529265201678 44.98910133229492) rotate(0 34.3564858840989 -1.9941705074807032)"><path d="M-2.42 1.6 C9.82 1.28, 59.41 -2.69, 70.86 -4 M1.46 0 C13.75 0.36, 58.07 0.36, 69.79 -0.37" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(166.03529265201678 44.98910133229492) rotate(0 34.3564858840989 -1.9941705074807032)"><path d="M-2.62 -12.21 C-3.38 -11.54, -4.37 -6.45, -1.54 1.31 M-3.21 -14.09 C-2.89 -7.41, -2.8 -2.51, -2.5 0.92" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(166.03529265201678 44.98910133229492) rotate(0 34.3564858840989 -1.9941705074807032)"><path d="M-0.51 17.71 C-1.48 12.13, -2.91 10.99, -1.54 1.31 M-1.09 15.83 C-1.25 11.75, -1.92 5.86, -2.5 0.92" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(166.03529265201678 44.98910133229492) rotate(0 34.3564858840989 -1.9941705074807032)"><path d="M67.85 -1.96 L57.96 5.5 L56.38 -5.96 L69.67 -0.01" stroke="none" stroke-width="0" fill="#1e1e1e" fill-rule="evenodd"></path><path d="M70.65 0.78 C66.94 -0.39, 62.93 2.98, 57.26 6.05 M70.07 -1.1 C65.24 2.72, 60.12 4.64, 56.29 5.65 M57.34 7.12 C55.77 2.64, 54.99 -1.52, 56.69 -7.36 M55.87 6.2 C56.15 2.66, 55.58 -1.04, 56.49 -6.18 M55.26 -6.82 C59.62 -5.77, 66.49 -3.82, 70.08 0.72 M55.67 -6.83 C59.48 -5.24, 61.5 -4.75, 70.04 0.15 M69.79 -0.37 C69.79 -0.37, 69.79 -0.37, 69.79 -0.37 M69.79 -0.37 C69.79 -0.37, 69.79 -0.37, 69.79 -0.37" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(331.52020450385044 45.51497919458052) rotate(0 41.330875948773155 -1.791108758676728)"><path d="M2.51 -0.98 C16.78 -1.83, 71.5 -4.01, 84.86 -4.8 M0.43 -3.98 C14.6 -4.48, 70.41 -1.87, 83.81 -1.83" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(331.52020450385044 45.51497919458052) rotate(0 41.330875948773155 -1.791108758676728)"><path d="M1.3 -14.79 C1.37 -11.91, 1.08 -8.33, 1.34 -0.02 M1.54 -16.55 C2.63 -10.24, 1.82 -4.43, 2.88 -0.9" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(331.52020450385044 45.51497919458052) rotate(0 41.330875948773155 -1.791108758676728)"><path d="M2.67 15.18 C2.34 9.93, 1.68 5.36, 1.34 -0.02 M2.91 13.42 C3.26 7.86, 1.9 1.8, 2.88 -0.9" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(331.52020450385044 45.51497919458052) rotate(0 41.330875948773155 -1.791108758676728)"><path d="M82.04 -1.18 L69.05 5.22 L70.02 -10.15 L82.13 -1.72" stroke="none" stroke-width="0" fill="#1e1e1e" fill-rule="evenodd"></path><path d="M83.29 -0.65 C79.42 -0.09, 75.21 1.03, 68.88 5.1 M83.53 -2.41 C78.84 0.42, 72.3 2.66, 70.42 4.22 M69.77 3.5 C70.09 0.64, 71.16 -5.67, 71.15 -8.68 M69.69 4.76 C69.92 1.58, 70.78 -0.62, 70.98 -8.5 M69.43 -8.94 C73.2 -5.63, 77.77 -5.51, 83.18 -2.59 M70.39 -8.87 C75.08 -6.12, 78.93 -4.25, 83.96 -1.42 M83.81 -1.83 C83.81 -1.83, 83.81 -1.83, 83.81 -1.83 M83.81 -1.83 C83.81 -1.83, 83.81 -1.83, 83.81 -1.83" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(505.63263732407813 52.83040886301478) rotate(0 67.95799859957015 0.5831516888715669)"><path d="M-0.24 -0.23 C22.27 0.09, 113.65 2.5, 136.49 2.69 M-3.82 -2.82 C18.25 -3.39, 111.35 -2.62, 134.56 -1.49" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(505.63263732407813 52.83040886301478) rotate(0 67.95799859957015 0.5831516888715669)"><path d="M-0.03 -14.92 C0.84 -9.51, -0.22 -5.74, -1.27 -0.75 M-0.04 -14.53 C-0.03 -11.3, 0.06 -7.78, -0.69 -0.61" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(505.63263732407813 52.83040886301478) rotate(0 67.95799859957015 0.5831516888715669)"><path d="M-0.7 15.07 C0.32 9.98, -0.5 3.24, -1.27 -0.75 M-0.71 15.46 C-0.62 11.76, -0.38 8.35, -0.69 -0.61" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(505.63263732407813 52.83040886301478) rotate(0 67.95799859957015 0.5831516888715669)"><path d="M134.34 -0.94 L121.17 4.69 L121.29 -8.78 L135.62 -2.03" stroke="none" stroke-width="0" fill="#1e1e1e" fill-rule="evenodd"></path><path d="M134.44 -1.18 C130.58 1.07, 124.83 1.68, 119.79 3.99 M134.43 -0.8 C131.31 0.31, 128.3 1.74, 120.37 4.13 M120.3 3.42 C120.71 0.63, 121.41 -1.64, 121.82 -8.4 M120.45 4.38 C121.26 1.58, 120.55 -1.15, 120.52 -8.06 M122.37 -7.6 C125.34 -5.36, 132.01 -2.55, 135.83 -2.72 M121.47 -7.63 C125.8 -6.1, 131.83 -2.9, 134.32 -0.95 M134.56 -1.49 C134.56 -1.49, 134.56 -1.49, 134.56 -1.49 M134.56 -1.49 C134.56 -1.49, 134.56 -1.49, 134.56 -1.49" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(573.3250936367513 54.16853372051446) rotate(0 -219.76227596470608 101.63500863190347)"><path d="M-0.33 -2.2 C-3.14 31.45, 54.5 169.05, -17.52 203.2 C-89.55 237.35, -362.37 232.13, -432.48 202.71 C-502.6 173.29, -436.99 56.57, -438.21 26.69 M-3.95 2.78 C-6.98 35.49, 51.31 165.21, -19.23 199 C-89.76 232.79, -357.54 235.12, -427.15 205.52 C-496.77 175.92, -434.19 51.76, -436.91 21.39" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(573.3250936367513 54.16853372051446) rotate(0 -219.76227596470608 101.63500863190347)"><path d="M13.39 -3.41 C11.27 -4.95, 6.06 -2.29, 0.39 -2.36 M13.86 -4.02 C11.35 -3.47, 7.76 -3.15, -0.56 -1.8" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(573.3250936367513 54.16853372051446) rotate(0 -219.76227596470608 101.63500863190347)"><path d="M-16.41 0.07 C-10.33 -2.4, -7.32 -0.7, 0.39 -2.36 M-15.94 -0.55 C-11.59 -0.65, -8.29 -1.13, -0.56 -1.8" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(573.3250936367513 54.16853372051446) rotate(0 -219.76227596470608 101.63500863190347)"><path d="M-438.02 21.67 L-433.98 36.47 L-446.96 33.14 L-436.25 20.63" stroke="none" stroke-width="0" fill="#1e1e1e" fill-rule="evenodd"></path><path d="M-438.09 21.91 C-435.09 23.94, -435.24 30.15, -432.74 35.82 M-437.63 21.3 C-435.79 24.91, -435.13 28.21, -433.69 36.38 M-433.52 35.69 C-437.46 36.04, -441.19 33.46, -445.97 33.88 M-433.01 35.88 C-437.56 35.63, -443.22 33.96, -445.7 32.81 M-445.84 32.57 C-444.05 27.8, -439.86 25.7, -437.59 22.04 M-445.99 33.37 C-442.72 29.24, -441.12 26.16, -436.24 21.71 M-436.91 21.39 C-436.91 21.39, -436.91 21.39, -436.91 21.39 M-436.91 21.39 C-436.91 21.39, -436.91 21.39, -436.91 21.39" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(13.374511243850066 41.19861535462951) rotate(0 40.638383318238084 2.520048369766755)"><path d="M0.05 0.88 C13.35 1.94, 69.18 5.6, 82.37 6.01 M-3.37 -1.11 C9.42 -0.89, 66.02 0.42, 80.73 1.53" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(13.374511243850066 41.19861535462951) rotate(0 40.638383318238084 2.520048369766755)"><path d="M1.5 -13.5 C1.15 -9.36, -0.65 -3.77, -1.03 2.12 M0.55 -14.34 C-0.11 -9.69, 0.94 -5.79, -0.09 0.54" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(13.374511243850066 41.19861535462951) rotate(0 40.638383318238084 2.520048369766755)"><path d="M-0.47 16.43 C-0.02 8.86, -1.05 2.76, -1.03 2.12 M-1.42 15.6 C-1.23 11.33, 0.41 6.3, -0.09 0.54" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g><g transform="translate(13.374511243850066 41.19861535462951) rotate(0 40.638383318238084 2.520048369766755)"><path d="M81.88 -0.12 L66.01 6.69 L65.51 -6.47 L80.04 -0.01" stroke="none" stroke-width="0" fill="#1e1e1e" fill-rule="evenodd"></path><path d="M81.2 2.12 C75.82 2.64, 68.97 4.61, 65.78 8.48 M80.25 1.28 C75.82 3.11, 73.03 4.24, 66.72 6.9 M66.21 6.82 C66.95 2.32, 68.42 -1.98, 67.54 -4.27 M67.34 6.95 C67.63 4.09, 67.39 1.53, 67.99 -5.5 M67.19 -4.5 C71.61 -5.39, 73.87 -3.52, 80.29 0.2 M67.08 -5.26 C73.3 -2.37, 77.59 -0.61, 80.91 2.09 M80.73 1.53 C80.73 1.53, 80.73 1.53, 80.73 1.53 M80.73 1.53 C80.73 1.53, 80.73 1.53, 80.73 1.53" stroke="#1e1e1e" stroke-width="1" fill="none"></path></g></g><mask></mask><g transform="translate(83.96710586635754 12.946998713404355) rotate(0 5 10)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">+</text></g><g transform="translate(157.2776038959273 87.59041488896673) rotate(0 3.287994384765625 10)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">-</text></g><g transform="translate(203.9713926977156 16.945753151381723) rotate(0 14.919990539550781 10)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">e(t)</text></g><g transform="translate(559.4752410260517 10) rotate(0 13.247993469238281 10)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#1e1e1e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">K/s</text></g>
</svg>

## improving transient response

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/transient-response-root-locus.webp]]

> [!question]
> How to calculate K?

- Product of distances from open-loop pole to point in question

> Second-order poles for the second-order system.

## improving steady state error (SSE)

adding PID (compensator) with an integrator ($\frac{1}{s}$) in feed forward path.

### ideal integral compensation

_proportional-plus-integral (PI) controller_
--> causing error to go to zero.
![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/ideal-integral-compensator.webp]]

> [!important]
> Add zero! on the pole near the origin at $s=-a$

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/zero-add-compensator.webp]]

$$
\frac{K}{s}(s+a) = K_p + \frac{K_i}{s}
$$

where $K_p$ is the proportional gain, and $K_i$ is the integral gain.

> [!important] Implementation $G_c(s)$
>
> $$
> G_c(s) = K_p + \frac{K_i}{s} = \frac{K_p(s+\frac{K_i}{K_p})}{s}
> $$

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/idea-integral-compensator-impl.webp]]

### lag compensation
