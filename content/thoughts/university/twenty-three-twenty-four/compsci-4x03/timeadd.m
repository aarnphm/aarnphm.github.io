function timeadd
% Define the sizes of the matrices
sizes = 500:100:1500;
times_addR = zeros(length(sizes), 1);
times_addC = zeros(length(sizes), 1);

% Time the functions and record the execution times
for i = 1:length(sizes)
  n = sizes(i);
  A = rand(n, n);
  B = rand(n, n);
  
  f_addR = @() addR(A, B);
  f_addC = @() addC(A, B);
  
  times_addR(i) = timeit(f_addR);
  times_addC(i) = timeit(f_addC);
end

% Perform least squares fitting to the model t = cn^2
X = [ones(length(sizes), 1), sizes'.^2];
crow_krow = X \ times_addR;
ccol_kcol = X \ times_addC;

% Output the constants
fprintf('crow: %e\n', crow_krow(1));
fprintf('krow: %e\n', crow_krow(2));
fprintf('ccol: %e\n', ccol_kcol(1));
fprintf('kcol: %e\n', ccol_kcol(2));

% Plot the results
figure;
loglog(sizes, times_addR, 'o-', 'DisplayName', 'addR');
hold on;
loglog(sizes, times_addC, 'o-', 'DisplayName', 'addC');
xlabel('Matrix Size (n)');
ylabel('Time (seconds)');
title('Time Complexity of Matrix Addition');
legend show;
grid on;
end

function C = addR(A, B)
[n, ~] = size(A);
C = zeros(n, n);
for i = 1:n
  C(i, :) = A(i, :) + B(i, :);
end
end

function C = addC(A, B)
[n, ~] = size(A);
C = zeros(n, n);
for j = 1:n
  C(:, j) = A(:, j) + B(:, j);
end
end
