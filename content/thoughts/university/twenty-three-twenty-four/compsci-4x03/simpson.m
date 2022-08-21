function I = simpson(f, a, b, n)
% Composite Simpson's Rule
% Ensure n is even
if mod(n, 2) == 1
  warning('Simpson’s rule requires an even number of intervals.');
  n = n + 1;
end
x = linspace(a, b, n+1); % Generate n+1 points from a to b
y = f(x);
dx = (b - a)/n;
I = (dx/3) * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
end
