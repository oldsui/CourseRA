function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);     %500
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

%X: 500*400

X_ = [ones(m, 1) X]';  %X_: 401*500 
Y = sigmoid(Theta1 * X_);       %Y:  25*500  
Y_ = [ones(1,size(Y,2)); Y];        %Y_:  26 * 500
Z = sigmoid(Y_' * Theta2');        % Z: 500*10
[prob, p] = max(Z, [], 2);


% =========================================================================


end
