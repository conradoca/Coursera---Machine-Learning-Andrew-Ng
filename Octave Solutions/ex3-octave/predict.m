function [Prob, p] = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% Use Theta1 to get into the hidden layer
% X size 5000x401, Theta1 size 25x401
% The resulting matrix will be 5000x25
HiddenLayer= sigmoid(X*Theta1');

% Add ones to the HiddenLayer
HiddenLayer = [ones(size(HiddenLayer, 1), 1) HiddenLayer];

% Use the HiddenLayer activation to calculate the output input_layer_size
% HiddenLayer is 5000x26, Theta2 is 10x26
OutputLayer= sigmoid(HiddenLayer*Theta2');

[Prob, p] = max(OutputLayer, [], 2);



% =========================================================================


end
