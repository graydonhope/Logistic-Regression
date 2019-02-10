function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% You need to return the following variables correctly 
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
  
  regularized_sum_value = 0;
  % Convention to not regularize over theta(1).
  for j = 2:num_theta_rows
    regularized_sum_value += theta([j]) ** 2;
  endfor
  lambda_over_m = (lambda / (2 * m)) * regularized_sum_value;
  new_theta_value = ((1 / m) * sum_value) + lambda_over_m;
  J = new_theta_value;  
  
  new_grad_value = (1 / m) * gradient_value;

  if (k == 1)
    gradient([k]) = new_grad_value;
  else 
    gradient([k]) = new_grad_value + ((lambda / m) * theta([k]));
  endif
  
endfor

grad = gradient;

end
