function nlsrun
%NLSRUN
%
% Set up data for neural net test
% Call nonlinear least squares optimization code to minimze the error
% Visualize results
%
% C F Higham and D J Higham, August 2017
%
%%%%%%% DATA %%%%%%%%%%%
% coordinates and targets
% Load the data
load points.mat x; % This loads 'x' which contains points and 'labels'
load points.mat labels; % This loads 'x' which contains points and 'labels'
x1 = x(1, :);
x2 = x(2, :);
y = labels;

figure(1)
clf
a1 = subplot(1,1,1);
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
hold on
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
a1.XTick = [0 1];
a1.YTick = [0 1];
a1.FontWeight = 'Bold';
a1.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_xy.png

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights/biases and call optimizer
rng(5000);
Pzero = 0.5*randn(23,1);

[finalP,finalerr] = lsqnonlin(@neterr,Pzero);

% Check this function
finalW2 = zeros(2,2);
finalW3 = zeros(3,2);
finalW4 = zeros(2,3);
finalW2(:) = finalP(1:4);
finalW3(:) = finalP(5:10);
finalW4(:) = finalP(11:16);
finalb2 = finalP(17:18);
finalb3 = finalP(19:21);
finalb4 = finalP(22:23);

% grid plot
N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = [0:Dx:1];
yvals = [0:Dy:1];
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activate(xy,finalW2,finalb2);
        a3 = activate(a2,finalW3,finalb3);
        a4 = activate(a3,finalW4,finalb4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
     end
end
[X,Y] = meshgrid(xvals,yvals);


figure(2)
clf
a2 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(X,Y,Mval,[0.5 0.5])
hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy.png

function costvec = neterr(Pval)
% return a vector whose two-norm squared is the cost function
   disp('new call')

   W2 = zeros(2,2);
   W3 = zeros(3,2);
   W4 = zeros(2,3);
   W2(:) = Pval(1:4);
   W3(:) = Pval(5:10);
   W4(:) = Pval(11:16);
   b2 = Pval(17:18);
   b3 = Pval(19:21);
   b4 = Pval(22:23);

   costvec = zeros(10,1);
   for i = 1:10
       a2 = activate(x,W2,b2);
       a3 = activate(a2,W3,b3);
       a4 = activate(a3,W4,b4);
       costvec(i) = norm(y(:,i) - a4,2);
   end
end % of nested function

function y = activate(x,W,b)
%ACTIVATE  Evaluates sigmoid function.
%
%  x is the input vector,  y is the output vector
%  W contains the weights, b contains the shifts
%
%  The ith component of y is activate((Wx+b)_i)
%  where activate(z) = 1/(1+exp(-z))

y = 1./(1+exp(-(W*x+b)));
end


end

