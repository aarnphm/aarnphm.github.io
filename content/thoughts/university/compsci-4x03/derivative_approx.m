function derivative_approx()

    % Define the function f and its derivative
    f = @(x) exp(sin(x));
    df = @(x) cos(x) * exp(sin(x));

    % Define the approximation functions g1 and g2
    g1 = @(x, h) (f(x + 2*h) - f(x)) / (2*h);
    g2 = @(x, h) (f(x + h) - f(x - h)) / (2*h);

    % Define x0
    x0 = pi/4;

    % Define k values and compute h values
    k_values = 1:0.5:16;
    h_values = 10.^(-k_values);

    % Initialize error arrays
    errors_g1 = zeros(size(h_values));
    errors_g2 = zeros(size(h_values));

    % Compute errors for each h_value
    for i = 1:length(h_values)
        h = h_values(i);
        errors_g1(i) = abs(df(x0) - g1(x0, h));
        errors_g2(i) = abs(df(x0) - g2(x0, h));
    end

    % Find the h value for which the error is the smallest for each approximation
    [~, idx_min_error_g1] = min(errors_g1);
    [~, idx_min_error_g2] = min(errors_g2);
    h_min_error_g1 = h_values(idx_min_error_g1);
    h_min_error_g2 = h_values(idx_min_error_g2);

    % Display the h values for the smallest errors
    fprintf('For g1, the smallest error is at h = %e\n', h_min_error_g1);
    fprintf('For g2, the smallest error is at h = %e\n', h_min_error_g2);

    % Plot errors using loglog
    loglog(h_values, errors_g1, '-o', 'DisplayName', '|f''(x_0) - g_1(x_0, h)|');
    hold on;
    loglog(h_values, errors_g2, '-x', 'DisplayName', '|f''(x_0) - g_2(x_0, h)|');
    hold off;

    % Add labels, title, and legend
    xlabel('h');
    ylabel('Error');
    title('Errors in Derivative Approximations');
    legend;
    grid on;

end