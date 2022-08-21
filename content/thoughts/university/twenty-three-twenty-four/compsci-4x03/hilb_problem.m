function hilb_problem()
    n = 1;
    while true
        % Generate Hilbert matrix of order n
        H = hilb(n);

        % Generate random vector x
        x = rand(n, 1);

        % Compute b = Hx
        b = H * x;

        % Solve the system Hx = b
        x_hat = H \ b;

        % Compute the relative error
        error = norm(x_hat - x) / norm(x);
        fprintf("error=%d, n=%d\n", error, n)
        % If the error is 100 percent, break
        if error >= 1
            break;
        end

        n = n + 1;
    end

    fprintf('\n=============\n\nThe largest n before the error is 100 percent is: %d\n\n=============\n', n-1);

    for i = 1:n-1
        H = hilb(i);
        x = rand(i, 1);
        b = H * x;
        x_hat = H \ b;

        r = b - H * x_hat;
        rel_resid = norm(r) / norm(b);
        rel_error = norm(x_hat - x) / norm(x);

        %fprintf('%d %.16f\n',i, rel_resid)
        fprintf('| %d | %.32f | %.32f |\n', i, rel_resid, rel_error);
    end

    cond_num = cond(H);
    fprintf('The condition number of the matrix for n = %d is: %f\n', n-1, cond_num);
end
