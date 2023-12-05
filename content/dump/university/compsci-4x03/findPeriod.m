function period = findPeriod(file_name)
    % Parse the data from the file
    data = parse(file_name);

    % Extract time, x, and y coordinates
    t = data(:, 1);
    x = data(:, 2);
    y = data(:, 3);

    % Define a tolerance for how close the object needs to be to its initial position
    tolerance = 1e-9;

    % Define interpolation functions for x and y, restricted to the range of data
    x_interp = @(tq) interp1(t, x, tq, 'spline', 'extrap');
    y_interp = @(tq) interp1(t, y, tq, 'spline', 'extrap');

    % Define the distance function from the initial position
    distance_from_initial = @(tq) sqrt((x_interp(tq) - x(1))^2 + (y_interp(tq) - y(1))^2);

    % Initial guess for fsolve - use the midpoint of the time data
    initial_guess = t(floor(length(t)/2));

    % Use fsolve to find the time at which the distance is minimized
    options = optimoptions('fsolve', 'Display', 'iter', 'TolFun', tolerance, 'MaxFunEvals', 10000);
    t_period = fsolve(distance_from_initial, initial_guess, options);

    % Calculate the period
    period = t_period - t(1);
end

function data = parse(file_name)
    % Open the file
    fid = fopen(file_name, 'rt');
    if fid == -1
        error('Failed to open file: %s', file_name);
    end

    % Read the data from the file
    % Assuming the data is separated by spaces or tabs
    data = fscanf(fid, '%f %f %f', [3, Inf]);

    % Transpose the data to have rows as individual entries
    data = data';

    % Close the file
    fclose(fid);
end
