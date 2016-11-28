function [w,b] = LDAFit2(X,Y)
%LDAFit2 
%   Function that takes as input 
%       X: a d x n matrix where each column corresponds to a feature vector
%          in R^d
%       Y: a 1 x n vector of binary labels (0,1) for each training vector
%   and generates an output of
%       w: a d x 1 normal vector
%       b: an offset for the separating hyperplane
%   using modified LDA (assuming Sigma = sigma*eye(d))

[d,n] = size(X);

idx0 = find(Y==0); % X(:,idx0) corresponds to all training vectors labeled 0
idx1 = find(Y==1); % X(:,idx1) corresponds to all training vectors labeled 1

n0 = length(idx0); % number of training vectors labeled 0
n1 = length(idx1); % number of training vectors labeled 1
n=n0+n1;

pi_hat_0 = n0/(n0+n1);
pi_hat_1 = n1/(n0+n1);

x1_0=X(:,idx0);
x2_0=X(:,idx1);

mu_hat_0 = mean(x1_0,2);
mu_hat_1 = mean(x2_0,2);

sum_matrix=zeros(d,d);

for i=1:n0
    
sum_matrix=sum_matrix + (x1_0(:,i)-mu_hat_0)*(x1_0(:,i)-mu_hat_0)';
end

for j=1:n1
    
 sum_matrix=sum_matrix + (x2_0(:,j)-mu_hat_1)*(x2_0(:,j)-mu_hat_1)';  
end    

Sigma_hat=(sum_matrix)/n ;
Sigma_hat=(1/d)*(trace(Sigma_hat)*eye(d))

w = inv(Sigma_hat)*(mu_hat_1-mu_hat_0);

b = (0.5*transpose(mu_hat_0)*inv(Sigma_hat)*mu_hat_0) - (0.5*transpose(mu_hat_1)*inv(Sigma_hat)*mu_hat_1) + log(pi_hat_0/pi_hat_1)
end




