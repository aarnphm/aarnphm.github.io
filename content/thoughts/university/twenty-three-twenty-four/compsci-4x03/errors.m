f = @(x) exp(x) .* cos(x);
a = 0;
b = pi/2;
n_values = 2:200; % n can be any integer for the trapezoidal rule
tol = 1e-4;
exact_integral = integral(f, a, b);

% Initialize arrays to store the actual errors and theoretical error bounds
actual_errors_trap = zeros(size(n_values));
bounds_trap = zeros(size(n_values));

% Compute the second derivative for the trapezoidal rule error bound
f_second = @(x) exp(x) .* (cos(x) - sin(x) - sin(x) - cos(x)); % f''(x)
max_f_second = max(abs(f_second(linspace(a, b, 1000)))); % Max over [a, b]

% Calculate errors and bounds for each n
for i = 1:length(n_values)
  n = n_values(i);
  
  % Trapezoidal rule calculations
  approx_integral_trap = trapezoid(f, a, b, n);
  actual_errors_trap(i) = abs(exact_integral - approx_integral_trap);
  bounds_trap(i) = ((b - a)^3 / (12 * n^2)) * max_f_second;
end

% Plot the error bounds and actual errors on a loglog plot
figure;
loglog(n_values, bounds_trap, 'r-', n_values, actual_errors_trap, 'b--');
legend('Trapezoid Bound', 'Trapezoid Actual');
title('Error Bounds and Actual Errors for Trapezoidal Rule');
xlabel('n (number of subintervals)');
ylabel('Error');

f = @(x) exp(x) .* cos(x);
a = 0;
b = pi/2;
n_values = 2:2:200; % Simpson's rule requires an even number of intervals
tol = 1e-4;
exact_integral = integral(f, a, b);

% Initialize arrays to store the actual errors and theoretical error bounds
actual_errors_simp = zeros(size(n_values));
bounds_simp = zeros(size(n_values));

% Compute the fourth derivative for Simpson's rule error bound
max_f_4th = max(abs(exp(linspace(a, b, 1000)) .* (cos(linspace(a, b, 1000)) - 4.*sin(linspace(a, b, 1000)) - 6.*cos(linspace(a, b, 1000)) - 4.*sin(linspace(a, b, 1000)) + cos(linspace(a, b, 1000)))));

% Calculate errors and bounds for each n
for i = 1:length(n_values)
  n = n_values(i);
  
  % Simpson's rule calculations
  approx_integral_simp = simpson(f, a, b, n);
  actual_errors_simp(i) = abs(exact_integral - approx_integral_simp);
  bounds_simp(i) = ((b - a)^5 / (180 * n^4)) * max_f_4th;
end

% Plot the error bounds and actual errors on a loglog plot
figure;
loglog(n_values, bounds_simp, 'r-', n_values, actual_errors_simp, 'b--');
legend('Simpson Bound', 'Simpson Actual');
title('Error Bounds and Actual Errors for Simpson''s Rule');
xlabel('n (number of subintervals)');
ylabel('Error');
