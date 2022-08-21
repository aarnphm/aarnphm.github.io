% Define the function f(x)
f = @(x) sin(x)./((1 + 20*x).^2);

% (a) Polynomial interpolation of degree n = 15 at equally spaced points

% Define the number of interpolation points and the degree of the polynomial
n = 15;
N = 100;

% Generate n+1 equally spaced points in the interval [-1, 1]
x = linspace(-1, 1, n+1);
y = f(x);

% Interpolate using polyfit
p_coeff = polyfit(x, y, n);

% Evaluate the interpolating polynomial at N equally spaced points
x_N = linspace(-1, 1, N);
p_N = polyval(p_coeff, x_N);

% Plot f(x) and p(x) on the same graph
figure;
plot(x_N, f(x_N), 'b-', x_N, p_N, 'r--', x, y, 'go');
legend('f(x)', 'p(x)', 'Interpolation Points');
title('f(x) and p(x) vs. x');
xlabel('x');
ylabel('y');

% Plot the absolute error |f(x) - p(x)| at the N points
figure;
plot(x_N, abs(f(x_N) - p_N), 'm-');
title('Absolute Error |f(x) - p(x)| vs. x');
xlabel('x');
ylabel('Error');

% (b) Polynomial interpolation using Chebyshev points

% Generate Chebyshev points in the interval [-1, 1]
x_cheb = cos((2*(1:n+1)-1)*pi/(2*n));
y_cheb = f(x_cheb);

% Interpolate using polyfit
p_cheb_coeff = polyfit(x_cheb, y_cheb, n);

% Evaluate the interpolating polynomial at N equally spaced points
p_cheb_N = polyval(p_cheb_coeff, x_N);

% Plot f(x) and p(x) using Chebyshev points on the same graph
figure;
plot(x_N, f(x_N), 'b-', x_N, p_cheb_N, 'r--', x_cheb, y_cheb, 'go');
legend('f(x)', 'p(x) with Chebyshev', 'Interpolation Points');
title('f(x) and p(x) with Chebyshev vs. x');
xlabel('x');
ylabel('y');

% Plot the absolute error |f(x) - p(x)| using Chebyshev points at the N points
figure;
plot(x_N, abs(f(x_N) - p_cheb_N), 'm-');
title('Absolute Error |f(x) - p(x) with Chebyshev| vs. x');
xlabel('x');
ylabel('Error');

% (c) Spline interpolation at n+1 equally spaced points

% Evaluate the function at n+1 equally spaced points
y_spline = f(x);

% Use the spline function to get the piecewise polynomial representation
pp = spline(x, y_spline);

% Evaluate the spline at N equally spaced points
spline_N = ppval(pp, x_N);

% Plot f(x) and the spline on the same graph
figure;
plot(x_N, f(x_N), 'b-', x_N, spline_N, 'r--', x, y_spline, 'go');
legend('f(x)', 'spline(x)', 'Interpolation Points');
title('f(x) and spline(x) vs. x');
xlabel('x');
ylabel('y');

% Plot the absolute error |f(x) - spline(x)| at the N points
figure;
plot(x_N, abs(f(x_N) - spline_N), 'm-');
title('Absolute Error |f(x) - spline(x)| vs. x');
xlabel('x');
ylabel('Error');
