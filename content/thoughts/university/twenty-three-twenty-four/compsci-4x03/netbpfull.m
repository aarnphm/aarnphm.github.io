function netbp_full
%NETBP_FULL
%   Extended version of netbp, with more graphics

% Load the data
load points.mat x; % This loads 'x' which contains points and 'labels'
load points.mat labels; % This loads 'x' which contains points and 'labels'

x_mean = mean(x, 2);
x_std = std(x, 0, 2);
x = (x - x_mean) ./ x_std; % Normalize the data

% Initialize weights and biases for a network with three outputs
rng(5000);
num_hidden_1 = 20; % Increased the number of neurons
num_hidden_2 = 20;
W2 = randn(num_hidden_1, 2) * 0.01;
W3 = randn(num_hidden_2, num_hidden_1) * 0.01;
W4 = randn(size(labels, 1), num_hidden_2) * 0.01;
b2 = zeros(num_hidden_1, 1);
b3 = zeros(num_hidden_2, 1);
b4 = zeros(size(labels, 1), 1);

% Training parameters
eta = 0.001; % Adjusted learning rate
alpha = 0.89; % Momentum term
alpha_leak = 0.01; % Define this once at the beginning of your script
lambda = 0.001; % L2 Regularization strength
Niter = 1e6;
batch_size = 16; % Adjusted batch size for batch training
% Learning rate decay
decay_rate = 0.99;
decay_step = 10000; % Apply decay every 10000 iterations

% buffers
savecost = zeros(Niter, 1);
saveaccuracy = zeros(Niter, 1);
savemisclassified = cell(Niter, 1);

% Momentum variables
mW2 = zeros(size(W2));
mW3 = zeros(size(W3));
mW4 = zeros(size(W4));
mb2 = zeros(size(b2));
mb3 = zeros(size(b3));
mb4 = zeros(size(b4));

% Training loop with batch training
for counter = 1:Niter
    % Select a batch of points
    batch_indices = randperm(size(x, 2), batch_size);
    x_batch = x(:, batch_indices);
    labels_batch = labels(:, batch_indices);

    % Initialize gradients for the batch
    gradW2 = zeros(size(W2));
    gradW3 = zeros(size(W3));
    gradW4 = zeros(size(W4));
    gradb2 = zeros(size(b2));
    gradb3 = zeros(size(b3));
    gradb4 = zeros(size(b4));

    % Loop over all examples in the batch
    for k = 1:batch_size
        xk = x_batch(:, k);
        labelk = labels_batch(:, k);

        % Forward pass
        a2 = actfn(xk, W2, b2, 'leaky_relu');
        a3 = actfn(a2, W3, b3, 'leaky_relu');
        a4 = actfn(a3, W4, b4, 'sigmoid');

        % Backward pass
        delta4 = (a4 - labelk) .* a4 .* (1 - a4);
        delta3 = (W4' * delta4) .* (a3 > 0 + alpha_leak * (a3 <= 0)); % Leaky ReLU derivative
        delta2 = (W3' * delta3) .* (a2 > 0 + alpha_leak * (a2 <= 0)); % Leaky ReLU derivative

        % Accumulate gradients over the batch
        gradW4 = gradW4 + delta4 * a3';
        gradW3 = gradW3 + delta3 * a2';
        gradW2 = gradW2 + delta2 * xk';
        gradb4 = gradb4 + delta4;
        gradb3 = gradb3 + delta3;
        gradb2 = gradb2 + delta2;
    end

    % Average gradients over the batch
    gradW4 = gradW4 + (lambda / batch_size) * W4;
    gradW3 = gradW3 + (lambda / batch_size) * W3;
    gradW2 = gradW2 + (lambda / batch_size) * W2;
    gradb4 = gradb4 / batch_size;
    gradb3 = gradb3 / batch_size;
    gradb2 = gradb2 / batch_size;

    % Update weights with gradients
    mW4 = alpha * mW4 - eta * gradW4;
    mW3 = alpha * mW3 - eta * gradW3;
    mW2 = alpha * mW2 - eta * gradW2;
    mb4 = alpha * mb4 - eta * gradb4;
    mb3 = alpha * mb3 - eta * gradb3;
    mb2 = alpha * mb2 - eta * gradb2;

    W4 = W4 + mW4;
    W3 = W3 + mW3;
    W2 = W2 + mW2;
    b4 = b4 + mb4;
    b3 = b3 + mb3;
    b2 = b2 + mb2;

    % Calculate cost and accuracy for the whole dataset
    [newcost, accuracy, misclassified] = cost(W2, W3, W4, b2, b3, b4, x, labels);
    savecost(counter) = newcost;
    saveaccuracy(counter) = accuracy;
    savemisclassified{counter} = misclassified;

    % Apply decay to the learning rate
    if mod(counter, decay_step) == 0
        eta = eta * decay_rate;
    end

    % Early stopping if accuracy is above 95%
    if accuracy >= 95
        fprintf('Achieved 95%% accuracy at iteration %d\n', counter);
        break;
    end

    if mod(counter, 10000) == 0 % Display cost and accuracy every 10000 iterations
        fprintf('iter=%d, cost=%e, accuracy=%.2f%%\n', counter, newcost, accuracy);
    end
end

% After training loop: Plot accuracy vs. number of iterations
figure;
plot(saveaccuracy);
xlabel('Number of Iterations');
ylabel('Accuracy (%)');
title('Accuracy vs. Number of Iterations');

% Plot cost vs. number of iterations
figure;
plot(savecost);
xlabel('Number of Iterations');
ylabel('Cost');
title('Cost vs. Number of Iterations');

% Plot decision boundaries and points
% First, create a meshgrid to cover the input space
[xv, yv] = meshgrid(linspace(min(x(1,:)), max(x(1,:)), 100), linspace(min(x(2,:)), max(x(2,:)), 100));
mesh_x = [xv(:)'; yv(:)'];
mesh_a2 = actfn(mesh_x, W2, b2, 'leaky_relu');
mesh_a3 = actfn(mesh_a2, W3, b3, 'leaky_relu');
mesh_a4 = actfn(mesh_a3, W4, b4, 'sigmoid');
[~, mesh_classes] = max(mesh_a4);
mesh_classes = reshape(mesh_classes, size(xv));

% Find the misclassified points from the last iteration
misclassified_indices = savemisclassified{end};
classified_correctly_indices = setdiff(1:size(x, 2), misclassified_indices);

% First Plot: Decision boundaries and correctly classified points only
figure;
contourf(xv, yv, mesh_classes);
hold on;
gscatter(x(1,classified_correctly_indices), x(2,classified_correctly_indices), vec2ind(labels(:,classified_correctly_indices)), 'rgb', 'osd', 12, 'LineWidth', 4);
title('Decision Boundaries and Correctly Classified Points');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
hold off;

% Second Plot: Decision boundaries and misclassified points only
figure;
contourf(xv, yv, mesh_classes);
hold on;
gscatter(x(1,misclassified_indices), x(2,misclassified_indices), vec2ind(labels(:,misclassified_indices)), 'rgb', 'osd', 12, 'LineWidth', 4);
title('Decision Boundaries and Misclassified Points Only');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Misclassified');
hold off;


% Activation function with switch for ReLU
function z = actfn(x, W, b, activation_type)
    if strcmp(activation_type, 'leaky_relu')
        % Define the Leaky ReLU slope for negative inputs
        alpha_leak = 0.01;
        z = max(alpha_leak * (W * x + b), W * x + b);
    elseif strcmp(activation_type, 'relu')
        z = max(0, W * x + b);
    else
        z = 1 ./ (1 + exp(-W * x - b));
    end
end

% Cost function with accuracy and misclassified indices calculation
function [costval, accuracy, misclassified] = cost(W2, W3, W4, b2, b3, b4, x, labels)
    misclassified = [];
    correct_count = 0;
    costval = 0; % Initialize the cost value

    for i = 1:size(x, 2)
        input = x(:, i);
        target = labels(:, i);
        a2 = actfn(input, W2, b2, 'leaky_relu');
        a3 = actfn(a2, W3, b3, 'leaky_relu');
        a4 = actfn(a3, W4, b4, 'sigmoid');

        % Compute the cross-entropy loss
        epsilon = 1e-12; % since it could happen log(0), so set a small epsilon
        costval = costval - sum(target .* log(a4 + epsilon) + (1 - target) .* log(1 - a4 + epsilon));

        [~, predicted_class] = max(a4);
        actual_class = find(target == 1);
        if predicted_class == actual_class
            correct_count = correct_count + 1;
        else
            misclassified = [misclassified, i];
        end
    end
    costval = costval / size(x, 2); % Average the cost over all examples
    accuracy = (correct_count / size(x, 2)) * 100;
end

end
