
%% Linear Regression with Gradient Descent without Regularization

kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = load('sleepdata_sanitized_three.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
m = ceil(n/10);
k = 1:m:n-m;
testingData = data_rand(k:k+m-1,:);
trainingData = [data_rand(1:k-1,:); data_rand(k+m:end,:)];

X = trainingData(:, 1:6);
y = trainingData(:, 7);
m = length(y);  %Number of training examples
d = size(X,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;


% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version % of X, where we substract the mean form each feature,
% then scale so that std dev = 1
[X, mu, stddev] = featureNormalize(X);

% Add intercept term to X
X = [ones(m,1) X];

% Init Theta and Run Gradient Descent
[theta, J_history] = gradientDescentMulti(X, y,theta, alpha, num_iters);

sq = predictSleepQuality(theta, testingData, mu, stddev);

w = ((X'*X)\X')*y;
avgSqErr=sum((y-X*theta).^2)./length(X)

avgDevErr=sum(abs(y-X*theta))./length(X)
figure;
plot(y,X*theta,'+');
title(sprintf('avgSqErr=%6.4f avDevErr=%6.4f',avgSqErr,avgDevErr));


avgSqErr=sum((testingData(:,7)-testingData(:,1:6).*theta(1:6,1)).^2)./length(testingData)

avgDevErr=sum(abs(testingData(:,7)-testingData(:,1:6)*theta(1:6,1)))./length(testingData)
figure;
plot(testingData(:,7),testingData(:,1:6)*theta(1:6,1),'+');
title(sprintf('avgSqErrTest=%6.4f avDevErrTest=%6.4f',avgSqErr,avgDevErr));




Theta = ((X'*X)\(X'))*y; % Normal Equations Method xaxis = 1:400;
figure;
plot(1:num_iters,J_history,'LineWidth',3)
title('Cost over 400 iterations')
xlabel('Number of Iterations'), ylabel('Cost'); grid on;

%% Linear Regression with Gradient Descent with Regularization

kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality


% WakeUpClass	Start	End	DurationInHourMin	DurationInMins	HeartRate	Activity(Steps)	SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
testingData = data_rand(1:n-ceil(n*.85),:);
trainingData = data_rand(ceil(n*.85)+1:n,:);


X = trainingData(:, 1:6);
y = trainingData(:, 7);
m = length(y);  %Number of training examples
d = size(X,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;

xtrain = X;
ytrain = y;
xtest = testingData(:, 1:6);
ytest = testingData(:, 7);

% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version % of X, where we substract the mean form each feature,
% then scale so that std dev = 1
X = featureNormalize(X);
xtest = featureNormalize(xtest);

% Add intercept term to X
X = [ones(m,1) X];
xtest = [ones(length(xtest),1) xtest];
[B, FitInfo] = lasso(X, y, 'cv', 5);
ax = lassoPlot(B, FitInfo, 'PlotType', 'CV');
lambda = 100;

% Init Theta and Run Gradient Descent
[theta, J_history] = gradientDescentMultiReg(X, y,theta, alpha, num_iters, lambda);

fprintf('Regularized Gradient Descent: [%f,%f]\n',theta, J_history);

avgSqErr=sum((ytest-xtest*theta).^2)./length(X)

avgDevErr=sum(abs(ytest-xtest*theta))./length(X)
figure;
plot(ytest,xtest*theta,'+');
title(sprintf('avgSqErr=%6.4f avDevErr=%6.4f',avgSqErr,avgDevErr));

figure;
plot(1:num_iters,J_history,'LineWidth',3)
title('Cost over 400 iterations')
xlabel('Number of Iterations'), ylabel('Cost'); grid on;



%%

X = trainingData(:, [3 5 6]);
y = trainingData(:, 7);
m = length(y);  %Number of training examples
d = size(X,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;

xtrain = X;
ytrain = y;
xtest = testingData(:, [3 5 6]);
ytest = testingData(:, 7);

% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version % of X, where we substract the mean form each feature,
% then scale so that std dev = 1
X = featureNormalize(X);
xtest = featureNormalize(xtest);

% Add intercept term to X
X = [ones(m,1) X];
xtest = [ones(length(xtest),1) xtest];
[B, FitInfo] = lasso(X, y, 'cv', 5);
ax = lassoPlot(B, FitInfo, 'PlotType', 'CV');
lambda = 100;

% Init Theta and Run Gradient Descent
[theta, J_history] = gradientDescentMultiReg(X, y,theta, alpha, num_iters, lambda);

fprintf('Regularized Gradient Descent: [%f,%f]\n',theta, J_history);
% sq = predictSleepQuality(theta, testingData, mu, stddev);

avgSqErr=sum((ytest-xtest*theta).^2)./length(X)

avgDevErr=sum(abs(ytest-xtest*theta))./length(X)
figure;
plot(ytest,xtest*theta,'+');
title(sprintf('avgSqErr=%6.4f avDevErr=%6.4f',avgSqErr,avgDevErr));

figure;
plot(1:num_iters,J_history,'LineWidth',3)
title('Cost over 400 iterations')
xlabel('Number of Iterations'), ylabel('Cost'); grid on;














%% SVM

kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = load('sleepdata_sanitized_three.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
m = ceil(n/10);
k = 1:m:n-m;
testingData = data_rand(k:k+m-1,:);
trainingData = [data_rand(1:k-1,:); data_rand(k+m:end,:)];

Xtrain = trainingData(:, 1:3);
Xtrain = [Xtrain trainingData(:, 5:7)];
ytrain = trainingData(:,4);

Xtest = testingData(:, 1:3);
Xtest = [Xtest testingData(:, 5:7)];
ytest = testingData(:,4);
% options = statset('UseParallel',true);

Mdl1 = fitcecoc(Xtrain,ytrain,'Coding','onevsone');
TestPredictionSVM1 = predict(Mdl1,Xtest);
[confusionMatrixSVM1, order] = confusionmat(ytest,TestPredictionSVM1);
SVM_accuracy1 = sum(diag(confusionMatrixSVM1))/sum(sum(confusionMatrixSVM1));

% CVMdl = crossval(Mdl1);

% figure
% plotconfusion(ytest, TestPredictionSVM1);

m = length(ytrain);  %Number of training examples
d = size(Xtrain,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;

%% Perceptron Neural Network


%% PCA
kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels=data_rand(:,1);
data_rand_features= data_rand(:,2:7);
[X, mu, stddev] = featureNormalize(data_rand_features);
testData = X(1:ceil(n*.85),:);
trainData = X(ceil(n*.85)+1:n,:);

ytest = data_rand(1:ceil(n*.85), 7);
ytrain = data_rand(ceil(n*.85)+1:n,7);

[coeff, score, latent, tsquared, explained, mu] = pca(X);

% % Create randomized 90/10 training/testing set
% n = size(data, 1);
% data_rand = data(randperm(n),:);
% testingData = data_rand(1:ceil(n*.85),:);
% trainingData = data_rand(ceil(n*.85)+1:n,:);
% 
[U, S, V] = svd(coeff);
k = 3;
Ureduce = U(:,1:k);
% data_rand = Ureduce'*X';
% data_rand = data_rand';
% 
% ytest = labels(1:49,1);
% ytrain = labels(50:482,1);
% testingData = data_rand(1:49,:);
% trainingData = data_rand(50:482,:);
%     
% Mdl1 = fitcecoc(trainingData,ytrain,'Coding','onevsone');
% TestPredictionSVM1 = predict(Mdl1,testingData);
% [confusionMatrixSVM1, order] = confusionmat(ytest,TestPredictionSVM1);
% SVM_accuracy1 = sum(diag(confusionMatrixSVM1))/sum(sum(confusionMatrixSVM1));

    



