function q = dsimpson(f, a, b, c, d, tol)
  function qx = integrand_x(y)
    [qx, ~] = adsimpson(@(x) f(x, y), a, b, tol);
  end
[q, ~] = adsimpson(@(y) integrand_x(y), c, d, tol);
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


