function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J=(1/(2*m))*sum(((theta'*X')-y').^2) + (lambda/(2*m))*sum(theta(2:end).^2);

% Getting the gradient and adding the regularization factor only on the indexes above 2
% (Octave starts calculating matrix indexes from 1 and not from 0)
% grad = (1/m)*X'*(sigmoid(X*theta)-y);
% + ((lambda/m)*theta);
% grad(2:end) = grad(2:end) +  ((lambda/m)*theta(2:end));

grad = (1/m)*X'*((theta'*X')-y')';
% + ((lambda/m)*theta);
grad(2:end) = grad(2:end) +  ((lambda/m)*theta(2:end));









% =========================================================================

grad = grad(:);

end
