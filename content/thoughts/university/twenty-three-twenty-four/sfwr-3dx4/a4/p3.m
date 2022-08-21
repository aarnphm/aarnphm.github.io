% System matrices
A = [0 1 0 0; 0 0 -1 0; 0 0 0 1; 0 0 9.8 0];
B = [0; 1; 0; -1];
C = eye(4); % Assuming full state output for simplicity
D = [0; 0; 0; 0];

% Desired closed-loop poles
p_desired = [-2+1i, -2-1i, -5, -5.01];

% State feedback controller design
K = place(A, B, p_desired);

% Validate the closed-loop system
A_cl = A - B*K;
sys_cl = ss(A_cl, B, C, D);

% Step response of the closed-loop system
figure;
step(sys_cl);
title('Closed-Loop Step Response with State Feedback Controller');

% Check the closed-loop poles
cl_poles = eig(A_cl);

% Display the designed K matrix and closed-loop poles
disp('State feedback gain matrix K:');
disp(K);
disp('Closed-loop poles:');
disp(cl_poles);
