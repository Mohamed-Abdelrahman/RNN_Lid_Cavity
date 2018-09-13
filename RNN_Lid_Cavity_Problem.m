clear; clc; close all;
%% Import Data
tic;

load x_train.txt; 
load y_train.txt;  

load ind.mat;

x_train = x_train(index,:);
y_train = y_train(index,:);
N_train = 12; %# of training data
N_test = 4;   %# of testing data
x_test = x_train(N_train+1:end,:);
y_test = y_train(N_train+1:end,:);

x_train = x_train(1:N_train,:);
y_train = y_train(1:N_train,:);
y_train = y_train*1;
x_train = x_train'; %x_train to a vector
y_train = y_train'; %y_train to a vector
x_test = x_test';   %x_test to a vector
y_test = y_test';   %y_test to a vector

%% Preprocess
Epoch = 1000;      % number of EPOCHs: the number of times the algorithm sees the entire data set
alpha = 0.001;     % learning rate
lambda = 0.001;    % L-2 regularization parameter
n0 = 2;            % number of nodes in input layer
n1 = 1000;         % number of nodes in hidden layer
n2 = 2;            % number of nodes in output layer
U1 = sqrt(6)/sqrt(n0+n1);
U2 = sqrt(6)/sqrt(n1+n2);
w1 = rand(n1,n0)*2*U1-U1;
w2 = rand(n2,n1)*2*U2-U2;
b1 = zeros(n1,1);
b2 = zeros(n2,1);
dw1 = zeros(n1,n0);
dw2 = zeros(n2,n1);
db1 = zeros(n1,1);
db2 = zeros(n2,1);
L_train = zeros(Epoch,1);
L_train1 = 0;
L_train2 = 0;
L_test1 = 0;
L_test2 = 0;

y_pred = zeros(n2,N_train); 
%% Training
for iter = 1:Epoch
    w1 = w1-alpha*dw1;
    w2 = w2-alpha*dw2;
    b1 = b1-alpha*db1;
    b2 = b2-alpha*db2;

    dw1 = zeros(n1,n0);
    dw2 = zeros(n2,n1);
    db1 = zeros(n1,1);
    db2 = zeros(n2,1);
    
    for i = 1:N_train
        % forward pass
        x = x_train(:,i); %Get input from training set
        h = w1*x+b1;
        sigOutput = sigmoid(h);
        sigDerivative = sigmoid_Derivative(sigOutput);
        y = w2*sigOutput+b2; %Calculated output based on neural network
        o = y_train(:,i); %Get output from training set
        error = y-o 
        % backward pass
        dw1 = dw1+w2'*2*error.*sigDerivative*x';
        dw2 = dw2+2*error*sigOutput';
        db1 = db1+w2'*2*error.*sigDerivative;
        db2 = db2+2*error;
        
        if iter==Epoch
            L_train1 = L_train1+error(1)^2;
            L_train2 = L_train2+error(2)^2;
        end
        L_train(iter) = L_train(iter)+(error)'*(error);
        if iter==Epoch
            y_pred(:,i) = y;
        end
    end
    
    L_train(iter) = 1/N_train*L_train(iter)+lambda/2*sum(sum(w1.^2))+lambda/2*sum(sum(w2.^2));
    dw1 = dw1/N_train+lambda*w1;
    dw2 = dw2/N_train+lambda*w2;
    db1 = db1/N_train;
    db2 = db2/N_train;
    display(iter);
end
L_train1 = 1/N_train*L_train1;
L_train2 = 1/N_train*L_train2;
y1 = [y_train(1,:);y_pred(1,:);y_train(2,:);y_pred(2,:)];
%% testing
for i = 1:N_test
        % forward pass
        x = x_test(:,i);
        h = w1*x+b1;
        sigOutput = sigmoid(h);
        y = w2*sigOutput+b2;
        o = y_test(:,i);
        error = y-o;
        L_test1 = L_test1+error(1)^2;
        L_test2 = L_test2+error(2)^2;
end
L_test1 = 1/N_test*L_test1;
L_test2 = 1/N_test*L_test2;
toc;