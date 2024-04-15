function sin_approx()
    % Define the values of x
    x_values = [0.1, 0.5, 1.0];
    
    % Loop through each value of x to compute the errors
    for i = 1:length(x_values)
        x = x_values(i);
        
        % Calculate the approximation using the given terms of the Taylor series
        approx = x - x^3/factorial(3) + x^5/factorial(5);
        
        % Calculate the actual value of sin(x)
        actual = sin(x);
        
        % Calculate the absolute error
        abs_error = abs(approx - actual);
        
        % Calculate the relative error
        rel_error = abs_error / abs(actual);
        
        % Display the results for each x
        fprintf('For x = %f:\n', x);
        fprintf('Approximated value: %f\n', approx);
        fprintf('Actual value: %f\n', actual);
        fprintf('Absolute Error: %e\n', abs_error);
        fprintf('Relative Error: %e\n\n', rel_error);
    end
end
