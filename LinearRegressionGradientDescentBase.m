%% Linear Regression with Gradient Descent without Regularization
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
labels = data_rand(:,7);
data_rand_features = data_rand(:,1:6);
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);

ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);

m = length(ytrain);  %Number of training examples
d = size(xtrain,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;


% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version % of X, where we substract the mean form each feature,
% then scale so that std dev = 1
% X = featureNormalize(X);

% Add intercept term to X
xtrain = [ones(m,1) xtrain];
xtest = [ones(length(xtest),1) xtest];

% Init Theta and Run Gradient Descent
[theta, J_history] = gradientDescentMulti(xtrain, ytrain,theta, alpha, num_iters);

avgSqErr = sum((ytest-xtest*theta).^2)./length(xtest)

avgDevErr = sum(abs(ytest-xtest*theta))./length(xtest)
figure;
plot(ytest,xtest*theta,'+');
title(sprintf('Gradient Descent w/ All Features\n AvgSqErr=%2.2f%% AvgDevErr=%2.2f%%',avgSqErr*100,avgDevErr*100));
print -dpng LinearRegressionGradientDescentErr_AllFeatures.png
eachSqErr = (ytest-xtest*theta).^2;
h = hist(eachSqErr,0:.01:max(eachSqErr));


figure;
plot(0:.01:max(eachSqErr),h,'linewidth',3); grid on;

title(sprintf('Gradient Descent w/ All Features Error Histogram'));
xlabel('Average Square Error'); ylabel('# of Occurances');
print -dpng LinearRegressionGradientDescentErr_AllFeaturesHistogram.png

%%
% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels = data_rand(:,7);
data_rand_features = data_rand(:,1:6);
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);


ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);



xtrain = [ones(m,1) xtrain(:,2) xtrain(:,3) xtrain(:,4)];
xtest = [ones(length(xtest),1) xtest(:, 2) xtest(:, 3) xtest(:, 4)];
m = length(ytrain);  %Number of training examples
d = size(xtrain,2); % Number of features.
theta = zeros(d,1); % Initialize thetas to zero.


% xtest = [ones(length(xtest),1) xtest];

%%%%% Top 3 Features

M2= xtrain; 

% xtest = X(1:n-ceil(n*.9),:);

[theta, J_history] = gradientDescentMulti(xtrain, ytrain,theta, alpha, num_iters);


avgSqErr = sum((ytest-xtest*theta).^2)./length(xtest);

avgDevErr = sum(abs(ytest-xtest*theta))./length(xtest);
figure;
plot(ytest,xtest*theta,'+');
title(sprintf('Gradient Descent w/ Top 3 Features\n AvgSqErrTest=%2.2f%% AvgDevErrTest=%2.2f%%',avgSqErr*100,avgDevErr*100));

eachSqErr = (ytest-xtest*theta).^2;
h = hist(eachSqErr,0:.01:max(eachSqErr));


figure;
plot(0:.01:max(eachSqErr),h,'linewidth',3); grid on;

title(sprintf('Gradient Descent w/ Top 3 Features Error Histogram'));
xlabel('Average Square Error'); ylabel('# of Occurances');
print -dpng LinearRegressionGradientDescentErr_Top3FeaturesHistogram.png


figure;
plot(1:num_iters,J_history,'LineWidth',3)
title('Cost over 400 iterations')
xlabel('Number of Iterations'), ylabel('Cost'); grid on;
print -dpng LinearRegressionGradientDescent_Cost.png

