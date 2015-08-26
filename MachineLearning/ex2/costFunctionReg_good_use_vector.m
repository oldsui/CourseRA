function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

    %cost function:
    H = sigmoid(X*theta);
    L = y.* log(H) + (1-y).* log(1-H);
    temp = ones(1,m);
    J = temp * L;
    J = -J/m;
    
    %add regularization term:
    n = size(theta,1);
    theta_r = theta(2:end);
    J = J + lambda/2/m* (theta_r' * theta_r);
    
    %gradient:grad = zeros(size(theta));
    grad = ((H-y)' * X)./m;
    
    %add regulartion term:
    grad = grad + lambda/m * theta'; 
    grad(1,1) = grad(1,1) - lambda/m * theta(1,1);
    
    %correct dimension from 1*n to n*1:
    grad = grad';


% =============================================================

end
