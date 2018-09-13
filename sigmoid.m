function sigOutput = sigmoid(x)
%This is the sigmoid function used in the RNN
sigOutput = 1./(1+exp(-x));
end

function sigDerivative = sigmoid_Derivative(sigOutput)
%This is the sigmoid function used in the RNN
sigDerivative = sigOutput*(1-sigOutput);
end