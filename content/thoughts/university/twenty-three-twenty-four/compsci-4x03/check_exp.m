function check_exp()
    % Define x
    x = 0.1;
    
    % Calculate the approximation using the first 6 terms of the Taylor series
    approx = 1 + x + x^2/factorial(2) + x^3/factorial(3) + x^4/factorial(4) + x^5/factorial(5);
    
    % Calculate the actual value of e^x
    actual = exp(x);
    
    % Calculate the error
    error = abs(approx - actual);
    
    % Display the results
    fprintf('Approximated value: %f\n', approx);
    fprintf('Actual value: %f\n', actual);
    fprintf('Error: %e\n', error);
    
    % Check if the error is less than 10^-8
    if error < 10^-8
        disp('The approximation is accurate up to 10^-8.');
    else
        disp('The approximation is NOT accurate up to 10^-8.');
    end
end
