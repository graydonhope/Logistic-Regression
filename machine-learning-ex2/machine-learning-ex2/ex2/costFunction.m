function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


num_theta_rows = size(theta, 1);
num_theta_columns = size(theta, 2);
num_X_columns = size(X,2);
num_y_columns = size(y, 2);

% Temporary matrix to hold new values after computation
gradient = nan(num_theta_rows, num_theta_columns);

for k = 1:num_theta_rows
  sum_value = 0;
  gradient_value = 0;
  for i = 1:m
    X_row_value = X([i],[1:num_X_columns]);
    first_half_value  = ((-1 * y([i])) * log(sigmoid(X_row_value * theta)));
    second_half_value = ((1 - y([i])) * log(1 - sigmoid(X_row_value * theta)));
    
    sum_value += first_half_value - second_half_value;
    gradient_value += (sigmoid(X_row_value * theta) - y([i])) * X([i],[k]);
  endfor
  new_theta_value = (1 / m) * sum_value;
  J = new_theta_value;  
  
  new_grad_value = (1 / m) * gradient_value;
  gradient([k]) = new_grad_value;
endfor

grad = gradient;

end
