function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    % all_y is the 'one v all' transformation of y
    %  y is a vector, all_y is a matrix
    % (vector) (matrix)
    %      y    ALL_Y
    %      1    1 0 0
    %      2    0 1 0
    %      3    0 0 1
    %      1    1 0 0
    %      2    0 1 0
    %      2    0 1 0
    all_y = zeros(size(y));
    for i=1:num_labels
	all_y(:,i) = (y==i); % compute the transformation
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% FORWARD PROPAGATION %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1 = [ones(m, 1) X];        % add the bias unit to the input
a2 = sigmoid(a1 * Theta1'); % calculate the activation function for layer 2
a2 = [ones(m,1) a2];        % add the bias unit to layer 2
a3 = sigmoid(a2 * Theta2'); % calculate the activation function for layer 3

I20 = eye(size(Theta2, 2)); I20(1,1) = 0; % this is to avoid regularizing theta2_0
I10 = eye(size(Theta1, 2)); I10(1,1) = 0; % this is to avoid regularizing theta1_0

% JmK is the matrix of the cost for 'm' training examples and 'K' output classes
JmK = -1/m * ( all_y  .* log(a3) \          % y = 1
	+ (1-all_y ) .* log(1-a3));         % y = 0

% Calculate the regularization terms dependent on each layer's Thetas
% Remove the bias multipier (first column), from theta values before summing
reg1 = sum(sum(lambda/(2*m) * (Theta1 * I10) .* Theta1));  % penalty for large theta1's
reg2 = sum(sum(lambda/(2*m) * (Theta2 * I20) .* Theta2));  % penalty for large theta2's
% calculate the regularized cost
J = sum(sum(JmK)) + reg1 + reg2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% BACK PROPAGATION %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
delta3 = a3 - all_y;
Theta2_grad = 1/m * (delta3' * a2 + lambda * Theta2 * I20);

delta2 = (delta3 .* sigmoidGradient(a2 * Theta2')) * Theta2;
delta2 = delta2(:,2:end);
Theta1_grad = 1/m * (delta2' * a1 + lambda * Theta1 * I10);

%%%delta1 = (delta2 .* sigmoidGradient(a1 * Theta1')) * Theta1;
%%%delta1 = delta1(:,2:end);

% =========================================================================

% Unroll gradients
% WARNING - unrolled A has same dimensions as unrolled A' (the transpose)
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
