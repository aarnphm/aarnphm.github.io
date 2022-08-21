function I = trapezoid(f, a, b, n)
    % Composite Trapezoidal Rule
    x = linspace(a, b, n+1); % Generate n+1 points from a to b
    y = f(x);
    dx = (b - a)/n;
    I = (dx/2) * (y(1) + 2*sum(y(2:end-1)) + y(end));
end
