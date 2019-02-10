function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

if ismatrix(z)
  num_rows = size(z, 1);
  num_columns = size(z, 2);
  sigmoid_values = nan(num_rows, num_columns);
  
  for i = 1:num_rows
    for j = 1:num_columns
      sigmoid_values([i],[j]) = ( 1 / (1 + exp(-1 * z([i],[j]))) );
    endfor
  endfor
  g = sigmoid_values;
else
  g = (1 / (1 + exp(-1 * z)));
endif

end
