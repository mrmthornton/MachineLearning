function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of feature plus 1 for theta_0.
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

I0 = eye(n); I0(1,1) = 0; % this is to avoid regularizing theta_0

Hypothesis = sigmoid(X * theta);

J = -1/m * ( y' * log(Hypothesis) \           % y = 1
	+ (1-y') * log(1-Hypothesis)) \        % y = 0
	+ lambda/(2*m) * theta' * I0 * theta;  % penalty for large theta

grad = 1/m * X' * (Hypothesis - y) + lambda/m * I0 * theta;

% =============================================================

end
