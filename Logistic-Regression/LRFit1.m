function [w,b] = LRFit1(X,Y)
%LRFit1
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using logistic regression, where w,b are estimated using standard
%   gradient descent
%load('synthetic1.mat');
[d,n] = size(X);

Xtilde = [ones(1,n); X];

theta = zeros(d+1,1); % Initialize starting point to zero
k = 0; % Iteration counter

maxiter = 10^5; % Set maximum number of iterations
G = 1; % Initialize G to be nonzero to enter loop
alpha = 0.01;
tic
while ((k<=maxiter)&&(norm(G)>1e-6))
    
    h= theta'*Xtilde;
    for j = 1:n
        G = G + Xtilde(:,j)*(Y(1,j) - 1/(1 + exp(-1*h(j))));
    end
    theta = theta + alpha*G;
    
    if(theta==0)
       
        break;
    end
    
    k = k+1;    
end
t = toc;

disp(['Total iterations: ' num2str(k)]);
disp(['Total time: ' num2str(t) ' sec']);

b = theta( 1,1 );
w = theta( 2:end,1 );
end