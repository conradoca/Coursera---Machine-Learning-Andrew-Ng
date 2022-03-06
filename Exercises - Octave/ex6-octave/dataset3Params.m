function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% Selected values
values = [0.01 0.03 0.1 0.3 1 3 10 30]';
% values = [0.01 0.03]';

% Matrix keeping validation errors and the C and sigma value
m = length(values);
error_val = zeros(m^2, 3);

row = 0;
for i = 1:m
    for j = 1:m
        row +=1;
        error_val(row, 1) = values(i); % C value
        error_val(row, 2) = values(j); % Sigma value
        model = svmTrain(X, y, error_val(row, 1), @(x1, x2) gaussianKernel(x1, x2, error_val(row, 2)));
        predictions = svmPredict(model, Xval);
        error_val(row, 3) = mean(double(predictions ~= yval));
    end
end

[minError, irow]=min(error_val(:,3));
C = error_val(irow, 1);
sigma = error_val(irow, 2);


% =========================================================================

end
