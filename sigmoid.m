function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
%  激活函数，值总在0,1之间
g = 1.0 ./ (1.0 + exp(-z));
end
