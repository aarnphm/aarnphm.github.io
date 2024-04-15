% Define the transfer function
num = 300 * [1 100];
den = conv([1 1], conv([1 10], [1 40]));
G = tf(num, den);

% Generate frequency vector (in rad/s)
w = logspace(-2, 3, 1000);

% Compute magnitude and phase
[mag, phase, w] = bode(G, w);

% Asymptotic magnitude approximation
asymp_mag = zeros(size(w));
asymp_mag(w < 1) = 300 * 100 / (1 * 10 * 40);  % DC gain
asymp_mag(w >= 1 & w < 10) = 300 * 100 ./ (w(w >= 1 & w < 10) * 10 * 40);  % -20 dB/dec slope
asymp_mag(w >= 10 & w < 40) = 300 * 100 ./ (w(w >= 10 & w < 40).^2 * 40);  % -40 dB/dec slope
asymp_mag(w >= 40) = 300 * 100 ./ (w(w >= 40).^3);  % -60 dB/dec slope

% Asymptotic phase approximation
asymp_phase = zeros(size(w));
asymp_phase(w < 0.1) = 0;  % 0 deg
asymp_phase(w >= 0.1 & w < 1) = -45;  % -45 deg
asymp_phase(w >= 1 & w < 10) = -90;  % -90 deg
asymp_phase(w >= 10 & w < 40) = -180;  % -180 deg
asymp_phase(w >= 40) = -270;  % -270 deg

% Plot Bode diagram
figure;
subplot(2, 1, 1);
loglog(w, squeeze(mag));
hold on;
loglog(w, asymp_mag, '--');
ylabel('Magnitude');
title('Asymptotic Bode Plot');
grid on;

subplot(2, 1, 2);
semilogx(w, squeeze(phase));
hold on;
semilogx(w, asymp_phase, '--');
xlabel('Frequency (rad/s)');
ylabel('Phase (deg)');
grid on;

% Find the bandwidth
mag_db = 20*log10(squeeze(mag));
bandwidth = w(find(mag_db >= -3, 1, 'last'));
fprintf('The bandwidth of the system is %.2f rad/s.\n', bandwidth);
