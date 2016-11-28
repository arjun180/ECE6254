function [w,b] = LRFit2(X,Y)
%LRFit2
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using logistic regression, where w,b are estimated using Newton's
%   method

[d,n] = size(X);

Xtilde = [ones(1,n); X];

theta = zeros(d+1,1); % Initialize starting point to zero
k = 0; % Iteration counter

maxiter = 15; % Set maximum number of iterations
G = 1; % Initialize G to be nonzero to enter loop

tic
while ((k<=maxiter)&&(norm(G)>1e-6))
    temp1= zeros(d+1,d+1);
    temp= zeros(d+1,1);
    for i=1:n
    temp = temp + (Xtilde(:,i)*(Y(1,i)- 1/(1+exp(theta'*Xtilde(:,i)))));
    temp1 = temp1 + (Xtilde(:,i)*Xtilde(:,i)'*1/(1+exp(theta'*Xtilde(:,i)))*(1-1/(1+exp(theta'*Xtilde(:,i)))));
    end
    G=temp;
    % Hessian
    H = temp;
    alpha= 1e-8;
    % Update
    theta = theta+alpha*inv(temp1)*G;
    
    k = k+1;
    
end
t = toc;

disp(['Total iterations: ' num2str(k)]);
disp(['Total time: ' num2str(t) ' sec']);
b = theta( 1,1 );
w = theta( 2:end,1 );
