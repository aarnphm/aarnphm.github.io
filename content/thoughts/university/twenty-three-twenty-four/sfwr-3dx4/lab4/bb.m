s = tf('s');
G = 0.419 / s^2;

OS = 0.05;
Ts = 3;

zeta = -log(OS) / sqrt(pi^2 + log(OS)^2);
wn = 4 / (zeta * Ts);

% Plot the root locus
figure;
rlocus(G);
sgrid(zeta, wn);
axis([-3 3 -3 3]);

rltool(G);

C = 6.942 * (1+2.3*s)/1;
