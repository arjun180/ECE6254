function [w,b] = LRFit3(X,Y)
%LRFit3
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using logistic regression, where w,b are estimated using stochastic 
%   gradient descent

[d,n] = size(X);

Xtilde = [ones(1,n); X];

theta = zeros(d+1,1); % Initialize starting point to zero
k = 0; % Iteration counter

maxiter = 10^5; % Set maximum number of iterations
G = 1; % Initialize G to be nonzero to enter loop

tic
while ((k<=maxiter)&&(norm(G)>1e-6))
    
    %G=zeros(d+1,1);
    % Generate random integer between 1 and n
    j = randperm(n,1);
    
    % Gradient
     G = (Xtilde(:,j)*(Y(1,j)- 1/(1+exp(theta'*Xtilde(:,j)))));
    
   % Step size
    alpha = 0.8;
    
    % Update
    theta = theta - alpha*G;
    
    k = k+1;
    
end
t = toc;

disp(['Total iterations: ' num2str(k)]);
disp(['Total time: ' num2str(t) ' sec']);

b = theta(1,1 );
w = theta(2:end,1);

end