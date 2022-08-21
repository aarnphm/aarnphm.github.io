% Define the transfer function
s = tf('s');
G = 1 / (s*(s^2 + 4*s + 13));

% Plot the root locus
figure;
rlocus(G);
sgrid(0.2588, 0);

% Find the gain K for a damping ratio of 0.2588
[k, poles] = rlocfind(G);

% Display the gain value
fprintf('The gain K for a damping ratio of 0.2588 is: %.3f\n', k);
