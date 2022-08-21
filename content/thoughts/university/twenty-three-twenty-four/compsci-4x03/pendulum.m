function pendulum
% Define the range for x values
x_values = linspace(-0.99, 0.99, 200); % Adjust the number of points for smoothness
K_values = zeros(size(x_values));
evals = zeros(size(x_values));
tol = 1e-10;

% Define the integrand for the elliptic integral of the first kind
for i = 1:length(x_values)
  x = x_values(i);
  integrand = @(theta) 1 ./ sqrt(1 - x^2 .* sin(theta).^2);
  
  % Use adsimpson to integrate and capture the number of function evaluations
  [K_values(i), evals(i)] = adsimpson(integrand, 0, pi/2, tol);
end

% Plot K(x) versus x
figure;
plot(x_values, K_values);
title('Complete Elliptic Integral of the First Kind K(x) versus x');
xlabel('x');
ylabel('K(x)');

% Plot the number of function evaluations versus x
figure;
plot(x_values, evals);
title('Number of Function Evaluations versus x');
xlabel('x');
ylabel('Number of Function Evaluations');
end

function [q, nfun] = adsimpson(f, a, b, tol)
persistent recursion_depth nfun_internal;
if isempty(recursion_depth)
  recursion_depth = 0;
end
if isempty(nfun_internal)
  nfun_internal = 0;
end
recursion_depth = recursion_depth + 1;
nfun_internal = nfun_internal + 1; % Increment function evaluations

if recursion_depth > 1000 % Check recursion depth
  error('Maximum recursion depth exceeded.');
end

c = (a + b)/2;
h = b - a;
fa = f(a); fb = f(b); fc = f(c);
S = (h/6) * (fa + 4*fc + fb);

d = (a + c)/2; e = (c + b)/2;
fd = f(d); fe = f(e);
Sleft = (h/12) * (fa + 4*fd + fc);
Sright = (h/12) * (fc + 4*fe + fb);
S2 = Sleft + Sright;

if abs(S2 - S) < 15*tol
  q = S2 + (S2 - S)/15;
else
  mid = (a + b)/2;
  [q_left, nfun_left] = adsimpson(f, a, mid, tol/2);
  [q_right, nfun_right] = adsimpson(f, mid, b, tol/2);
  q = q_left + q_right;
  nfun_internal = nfun_internal + nfun_left + nfun_right;
end

if nargout > 1
  nfun = nfun_internal;
end

recursion_depth = recursion_depth - 1;
if recursion_depth == 0
  nfun_internal = 0; % Reset on the last exit
end
end
